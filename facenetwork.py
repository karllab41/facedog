import numpy as np
import pickle
import tensorflow as tf
import sys

# Read data files
def read_back2future(read_images=False):
    '''
    A function that will be taken in by FaceTuning.
    '''
    indices = pickle.load(open('BTTF1_N2I.pickle', 'rb'))
    data = pickle.load(open('DATASTORE_BTTF1.pickle', 'rb'))
    labels = list(indices.keys())
    if read_images:
        images = pickle.load(open('BTTF1.mp4.face_detected_big.pickle', 'rb'))
        return data, labels, indices, images
    return data, labels, indices

class FaceTuning:

    def __init__(self, filereader, inputsize=128, embedsize=128, network=[], graph=None, ownsession=True, verbose=True):
       
        '''
        Class: FaceTuning - Takes in a file reader and a network size. 

        The filereader to be passed in must return:
        - data
        - labels - list of names in the order. Can be strings. The label number will be the 
                   location index of the name.
        - indices - key value pair, where the key is the label name and the value are all
                    the indices of the input vectors that relate to that label
        '''
        self.data, self.labels, self.indices = filereader()
        self.revlabels = None
        self.graph = self.sampledlogcost(inputsize, embedsize, len(self.labels)+1)
        self.saver = tf.train.Saver()
        self.runningcost = []
        self.verbose = verbose
        
        if ownsession:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
            self.sess.run(tf.global_variables_initializer())

    def sampledlogcost(self, inputsize, embedsize, outputsize, network=[], batchsize=None, numindices=None):

        '''
        Creates a graph that implements the sampled logistic cost function
        off a two layer neural network, where the hidden layer as the 
        embedding.

           features -> Weights 0 ---(hidden-units)---> Weights 1 ---> cost

        Input:

           - inputsize: input feature size
           - embedsize: embedding size (how many hidden units)
           - outputsize: your label space size (only useful during training)

        Output:

           - dictionary of nodes in the graph with the following keys:
             - placeholder graph nodes used for inputs 
               > indices: specify the locations of the appropriate output vectors
                 to use in optimization. The size is ( batchsize, #samples )                  
               > labels: a matrix/list of size ( batchsize, #samples ) that specifies
                 whether or not we have a positive or negative sample with [+/- 1]
               > features: input features used as the input to the neural network
             - placeholder graph nodes used for outputs
               > hidden_units: when passing a feature into the neural network, this
                 results in the embedding vector, which after training will yield
                 the appropriate feature vector
               > cost: the cost function used for training
        '''
        indices = tf.placeholder(tf.int32, shape=(batchsize,numindices))
        labels  = tf.placeholder(tf.int32, shape=(batchsize,numindices))
        features= tf.placeholder(tf.float32, shape=(batchsize, inputsize))
        outvecs = tf.Variable( tf.truncated_normal((outputsize, embedsize), stddev=0.4) )

        network += [embedsize]

        # Connect core neural network
        weightin = inputsize
        infeats = features
        netmatrix = []
        for hidden in network:
            netmatrix = tf.Variable( tf.truncated_normal((weightin, hidden), stddev=0.4) )
            hidden_layer = tf.matmul(infeats, netmatrix)
            weightin = hidden
            infeats = hidden_layer

        # Sampling on the final layer
        hidden_layer_expanded = tf.expand_dims(hidden_layer, 1)
        indices_feed = tf.expand_dims(indices, 2)
        sampledvecs = tf.gather_nd( outvecs, indices_feed )

        # Final cost function
        dots = tf.reduce_sum(hidden_layer_expanded * sampledvecs, axis=2)
        logs = -tf.log( tf.sigmoid(dots) )
        unitcost = tf.reduce_sum(logs)
        totcost = tf.reduce_mean(unitcost)
        optimizer= tf.train.AdamOptimizer()
        opt = optimizer.minimize(totcost)

        return {'indices': indices, 'labels': labels, 'features': features,
                'hidden_layer': hidden_layer, 'cost': totcost, 'opt': opt}
    
    
    def reverse_labels(self,labels):

        '''
        Internal functionality to turn assign numbers to labels
        '''
        self.revlabels = dict()
        keys = np.linspace( 0, len(labels), len(labels)+1 ).astype(int)
        for key in keys[:-1]:
            self.revlabels[ labels[key] ] = key+1
        return self.revlabels
    
    
    def get_sample(self):
        '''
        Get a single sample from the dataset
        '''
        data = self.data
        labels = self.labels
        indices = self.indices
        revlabels = self.revlabels
        
        if not revlabels:
            revlabels = self.reverse_labels(labels)

        samps = np.random.choice(labels, 4, replace=False)
        dataidx = np.random.choice(list(indices[samps[0]]))
        feature = data[0][dataidx]

        negidx = np.zeros(4).astype(int)
        for i, name in enumerate(samps):
            negidx[i] = revlabels[name]

        posneg= -np.ones(4)
        posneg[0] = 1

        return feature, negidx, posneg, dataidx

    def get_batch(self,numsamps):
        
        data = self.data
        labels = self.labels
        indices = self.indices
        revlabels = self.revlabels

        if not revlabels:
            revlabels = self.reverse_labels(labels)

        features = np.zeros((numsamps, len(data[0][0])))
        sampidx = np.zeros((numsamps, 4)).astype(int)
        posnegs = np.zeros((numsamps, 4)).astype(int)
        dataidx = np.zeros(numsamps).astype(int)

        for i in range(numsamps):
            features[i], sampidx[i], posnegs[i], dataidx[i] = self.get_sample()

        return features, sampidx, posnegs, dataidx
    
    def backprop_batch(self, sess=None):
        '''
        Perform backpropagation on a single batch
        '''
        if not sess:
            sess = self.sess
        
        features, samples, posneg, dataidx = self.get_batch(100)
        feeddict = {self.graph['features']: features, self.graph['indices']: samples, 
                    self.graph['labels']: posneg }
        _, costsofar = sess.run( [self.graph['opt'], self.graph['cost']], feeddict)
        self.runningcost += [costsofar]
        
    def backprop_iters(self, numiter, sess=None):
        '''
        Perform backpropagation. 

        Arguments:
        - numiter - The number of iterations to backpropagate over
        - sess - Which session to use. If it's none, we'll just use the one we created
        ''' 
        for i in range(numiter):
            self.backprop_batch(sess=sess)
            if self.verbose:
                sys.stdout.write('\rCurrent Cost='+str(self.runningcost[-1]))
                
    def save_model(self, modelname, sess=None):        
        '''
        Save the model to disk

        Argument:
        - modelname: the filename where you want to save this model
        '''
        if not sess:
            sess = self.sess
        save_path = self.saver.save(sess, modelname)
        
    def load_model(self, modelname, sess=None):
        '''
        Load a model in with argument "modelname"
        ''' 
        if not sess:
            sess = self.sess
        self.saver.restore(sess, modelname)

    def get_embedding(self, inputfeat, sess = None):
        '''
        Forward propagate to the last layer before softmax
        - inputfeat = numsamps x featuresize
        '''
        if not sess:
            sess = self.sess

        # In case you only have a single vector, this will reshape
        inputfeatshape = inputfeat.shape
        if len(inputfeatshape)==1:
            inputfeatshape = inputfeatshape[0]
            inputfeat = inputfeat.reshape((1,inputfeatshape)) 

        feeddict = {self.graph['features']: inputfeat} 

        return sess.run( self.graph['hidden_layer'], feeddict ).squeeze()

# loonytune.get_sample()[0].reshape((1,128))
loonytune = FaceTuning(read_back2future, network=[128,128])

# Load the model in?
# loonytune.load_model('saved-model')
loonytune.backprop_iters(50)

# Save the model
# loonytune.save_model('saved-model-test')
               
new_embed_sample = loonytune.get_embedding( loonytune.get_sample()[0])
new_embed_batch  = loonytune.get_embedding( loonytune.get_batch(5)[0])

