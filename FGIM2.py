import tensorflow as tf
from aggregators_odcf import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class FGIM2(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation,adj_user, adj_item):
        self._parse_args(args, adj_entity, adj_relation,adj_user, adj_item)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, adj_user,adj_item):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.ui_iter = args.ui_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.adj_user = adj_user
        self.adj_item = adj_item
        self.ui_act = tf.nn.relu
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.drop_ratio = args.ratio
        self.neighbor_item = args.neighbor_item
        self.cap_K = args.cap_k
        self.routing_times = args.routing_times
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=FGIM2.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=FGIM2.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=FGIM2.get_initializer(), name='relation_emb_matrix')
        self.route_weights =dict()
        for k in range(self.cap_K):
            self.route_weights['Wu_cap_%d' % k] = tf.get_variable(shape=[self.dim, self.dim],
            initializer=FGIM2.get_initializer(), name='Wu_cap_%d'%k)
            self.route_weights['Wi_cap_%d' % k] = tf.get_variable(shape=[self.dim, self.dim],
            initializer=FGIM2.get_initializer(),name='Wi_cap_%d' % k)

        # [batch_size, dim]
        #self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.user_neighbor = self.get_user_neighbors(self.user_indices, self.adj_user) #[[N,1],[N,20]]
        self.user_embeddings = self.aggregate_user( self.user_neighbor)
        self.item_neighbor = self.get_item_neighbors(self.item_indices, self.adj_item)
        self.item_implicit_embeddings = self.aggregate_item(self.item_neighbor)
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        self.entities, self.relations = self.get_neighbors(self.item_indices)
        # [batch_size, dim]
        self.item_explicit_embeddings, self.aggregators = self.aggregate(self.entities, self.relations)
        self.item_embeddings = tf.add(self.item_explicit_embeddings, self.item_implicit_embeddings)
        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            shape_neighbor = pow(self.n_neighbor, (i+1))
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, shape_neighbor])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, shape_neighbor])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def get_user_neighbors(self, seeds, adj):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        for i in range(self.ui_iter):
            shape_neighbor = pow(self.neighbor_item, (i + 1))
            if i%2==0:
                neighbor_entities = tf.reshape(tf.gather(adj, entities[i]), [self.batch_size, shape_neighbor])
            elif i%2==1:
                neighbor_entities = tf.reshape(tf.gather(self.adj_item, entities[i]), [self.batch_size, shape_neighbor])
            entities.append(neighbor_entities)
        return entities

    def get_item_neighbors(self, seeds, adj):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        for i in range(self.ui_iter):
            shape_neighbor = pow(self.neighbor_item, (i + 1))
            if i % 2 == 0:
                neighbor_entities = tf.reshape(tf.gather(adj, entities[i]), [self.batch_size, shape_neighbor])
            elif i % 2 == 1:
                neighbor_entities = tf.reshape(tf.gather(self.adj_user, entities[i]), [self.batch_size, shape_neighbor])
            entities.append(neighbor_entities)
        return entities
    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, dropout=0, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector, self.z_kg = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    context = tf.expand_dims(self.user_embeddings,1),
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])
        return res, aggregators

    def aggregate_user(self, neighbor):
        vectors = []
        for i in range(len(neighbor)):
            if i%2==0:
                vectors.append(tf.nn.embedding_lookup(self.user_emb_matrix, neighbor[i]))
            else:
                vectors.append(tf.nn.embedding_lookup(self.entity_emb_matrix, neighbor[i]))
        for i in range(self.ui_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.neighbor_item, self.dim]
                vector, self.user_p, self.arg = self.DisenAggregator(self_vectors=vectors[hop],
                                                         neighbor_vectors=tf.reshape(vectors[hop+1],
                                                                                     shape),user =True)
                entity_vectors_next_iter.append(vector)
            vectors = entity_vectors_next_iter
        res = tf.reshape(vectors[0], [self.batch_size, self.dim])
        return res

    def aggregate_item(self, neighbor):
        vectors = []
        for i in range(len(neighbor)):
            if i%2==0:
                vectors.append(tf.nn.embedding_lookup(self.entity_emb_matrix, neighbor[i]))
            else:
                vectors.append(tf.nn.embedding_lookup(self.user_emb_matrix, neighbor[i]))
        for i in range(self.ui_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.neighbor_item, self.dim]
                vector, _, _= self.DisenAggregator(self_vectors=vectors[hop],
                                                         neighbor_vectors=tf.reshape(vectors[hop+1],
                                                                                     shape),user =False)
                entity_vectors_next_iter.append(vector)
            vectors = entity_vectors_next_iter
        res = tf.reshape(vectors[0], [self.batch_size, self.dim])
        return res

    def DisenAggregator(self, self_vectors, neighbor_vectors, user):
        m = neighbor_vectors.get_shape().as_list()[-2]
        multiple_embeddings = dict()
        self_vectors = tf.reshape(self_vectors, [-1, self.dim])
        neighbor_vectors = tf.reshape(neighbor_vectors, [-1, self.dim])
        for i in range(self.cap_K):
            if user:
                self_vectors = self.ui_act(tf.matmul(self_vectors, self.route_weights['Wu_cap_%d' % i]))
                neighbor_vectors = self.ui_act(tf.matmul(neighbor_vectors, self.route_weights['Wu_cap_%d' % i]))
            if user==False:
                self_vectors = self.ui_act(tf.matmul(self_vectors, self.route_weights['Wu_cap_%d' % i]))
                neighbor_vectors = self.ui_act(tf.matmul(neighbor_vectors, self.route_weights['Wu_cap_%d' % i]))
            multiple_embeddings['s_%d' % i] = tf.expand_dims(self_vectors, axis=1)
            multiple_embeddings['n_%d' % i] = tf.expand_dims(neighbor_vectors, axis=1)
            if (i == 0):
                self_embeddings = multiple_embeddings['s_%d' % i]
                neighbor_embeddings = multiple_embeddings['n_%d' % i]
            else:
                self_embeddings = tf.concat([self_embeddings, multiple_embeddings['s_%d' % i]], axis=1)
                neighbor_embeddings = tf.concat([neighbor_embeddings, multiple_embeddings['n_%d' % i]], axis=1)
        u_k = tf.nn.l2_normalize(tf.reshape(self_embeddings, [-1, self.cap_K, self.dim]), axis=2)
        v_k = tf.nn.l2_normalize(tf.reshape(neighbor_embeddings, [-1, m, self.cap_K, self.dim]), axis=3)
        z = tf.reduce_sum(tf.multiply(tf.expand_dims(u_k, axis=1), v_k), axis=3)
        p = tf.nn.softmax(z, axis=2)
        arg_max = tf.argmax(p, axis=2);
        z = tf.nn.l2_normalize(tf.reduce_sum(v_k * tf.expand_dims(p, axis=-1), axis=1), axis=2)  # [-1,k,d]
        c_k = tf.math.l2_normalize(tf.add(z, u_k), axis=2)
        beta = tf.nn.softmax(tf.reduce_sum(c_k, axis=2), axis=1)
        r_k = tf.nn.l2_normalize(tf.reduce_sum(tf.expand_dims(beta, axis=-1) * c_k, axis=1), axis=1)
        r_k = tf.nn.dropout(r_k, self.drop_ratio)
        return r_k, p, arg_max

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))
        for i in range(self.cap_K):
            for j in range(i+1, self.cap_K):
                if i==0 and j==1:
                    self.orthogonal_loss = tf.nn.l2_loss(tf.reduce_sum(tf.multiply(self.route_weights['Wu_cap_%d' % i], self.route_weights['Wu_cap_%d' % j]),axis=1))
                    # self.orthogonal_loss += tf.nn.l2_loss(tf.reduce_sum(
                    #     tf.multiply(self.route_weights['Wi_cap_%d' % i], self.route_weights['Wi_cap_%d' % j]),axis=1))
                else:
                    self.orthogonal_loss += tf.nn.l2_loss(tf.reduce_sum(tf.multiply(self.route_weights['Wu_cap_%d' % i], self.route_weights['Wu_cap_%d' % j]),axis=1))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix) + self.orthogonal_loss
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        for k in range(self.cap_K):
            self.l2_loss += tf.nn.l2_loss(self.route_weights['Wu_cap_%d' % k])
        self.loss = self.base_loss + self.l2_weight * self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
    def get_sne(self, sess, feed_dict):
        return sess.run([self.user_neighbor[1], self.user_p, self.item_implicit_embeddings, self.arg], feed_dict)
