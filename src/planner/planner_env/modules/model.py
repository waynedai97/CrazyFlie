import torch
import torch.nn as nn
import math
from .test_parameter import *
#import time

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention,self).__init__()
        # Intilaize the parameters
        self.n_heads = n_heads
        self.input_dim = embedding_dim    #128
        self.embedding_dim = embedding_dim #Divide the number of embedding dimension by number of heads
        self.query_dim = embedding_dim // n_heads #Returns multiples of n_heads without remainder
        self.key_dim = embedding_dim // n_heads
        self.value_dim = self.key_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)
        
        # Weight matrix that are tensors limited to the module
        self.query_weight = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.query_dim))   #No of matrix, rows, columns 8*128*16
        self.key_weight = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.value_weight = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.output_weight = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))
        
        self.init_parameters()
        
    def init_parameters(self):
        # Initialize the parameters
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1)) #-1 refer to the last dimension
            param.data.uniform_(-stdv, stdv)
            
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        """
        :param query: queries (batch_size, n_query, input_dim)
        :param key: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        n_batch, n_key, n_dim = key.size() #n_batch: batch size, n_key: number of nodes, n_dim: embedding dimension
        n_query = query.size(1)     #number of queries
        n_value = value.size(1)
        
        key_flat = key.contiguous().view(-1, n_dim)    #.view() returns same data with different shape, use -1 to get inferred shape
        value_flat = value.contiguous().view(-1, n_dim)  #360,128
        query_flat = query.contiguous().view(-1, n_dim)
        shape_value = (self.n_heads, n_batch, n_value, -1)
        shape_key = (self.n_heads, n_batch, n_key, -1)
        shape_query = (self.n_heads, n_batch, n_query, -1)
        
        Q = torch.matmul(query_flat, self.query_weight).reshape(shape_query)  # n_heads*batch_size*n_query*key_dim 8*1*360*16
        K = torch.matmul(key_flat, self.key_weight).reshape(shape_key)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(value_flat, self.value_weight).reshape(shape_value)  # n_heads*batch_size*targets_size*value_dim
        
        # Scaled dot product attention
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size


        if attn_mask is not None: #edge mask, 1 to mask
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)    #Broadcast to the shape of U

        if key_padding_mask is not None: # Vector repeated to nxn mask for node padding, to 360 as the node size might not be constant
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1) #repeat number of rows
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)             #To ensure that all 1 is obtained
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8) #if mask is greater than 0, -1e8 is the value to be filled
        
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim
        
        #permute to change the order of the dimensions   reshape: To get back full dimension embeddings
        mat1 = heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim) # batch_size*n_query*n_heads*value_dim for permutatuion 360*128
        weight_matrix = self.output_weight.view(-1, self.embedding_dim) # n_heads*value_dim*embedding_dim 128*128
        out = torch.mm(mat1, weight_matrix).view(-1, n_query, self.embedding_dim)  # batch_size*n_query*embedding_dim 1,360,128
        
        return out, attention
    
class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())     #-1 refer to last dimension   

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        query = self.normalization1(src)
        h, _ = self.multiHeadAttention(query=query, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + src
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2
    
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(query=tgt, key=memory, value=memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + tgt
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w
    
class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, input, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            input = layer(input, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return input


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w

# Pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.query_weight = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.key_weight = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, query, key, mask=None):

        n_batch, n_key, n_dim = key.size()
        n_query = query.size(1)

        k_flat = key.reshape(-1, n_dim)
        q_flat = query.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.query_weight).view(shape_q)
        K = torch.matmul(k_flat, self.key_weight).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)  #read 

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention

class PolicyNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PolicyNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim) #Enhanceed current node features plus current node features

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask): #Node padding mask to ensure is 360 nodes.
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(input=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return enhanced_node_feature

    def output_policy(self, enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask):
        k_size = edge_inputs.size()[2]
        current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, k_size)) #repeat the current index to get the current edge features
        current_edge = current_edge.permute(0, 2, 1) #rearrange the dimension
        embedding_dim = enhanced_node_feature.size()[2] #Last dimension of the enhanced node feature

        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))
        #To form new matrix from enhanced node feature 
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))

        if edge_padding_mask is not None:
            current_mask = torch.gather(edge_padding_mask, 1, current_index.repeat(1,1,k_size)).to(enhanced_node_feature.device)    #repeat current index k times
        else:
            current_mask = None
        current_mask[:,:,0] = 1 # don't stay at current position
        #print('edge_padding_mask', edge_padding_mask, edge_padding_mask.size())
        #print('current index', current_index)
        #print('cuurent index' , current_index.repeat(1,1,k_size))
        #print('current mask', current_mask)
        #assert 0 in current_mask

        enhanced_current_node_feature, _ = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        enhanced_current_node_feature = self.current_embedding(torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1))
        logp = self.pointer(enhanced_current_node_feature, neigboring_feature, current_mask)
        logp= logp.squeeze(1) # batch_size*k_size, remove all dimension value of 1

        return logp

    def forward(self, node_inputs, edge_inputs, current_index, node_padding_mask=None, edge_padding_mask=None, edge_mask=None):
        #t1 = time.time()
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        logp = self.output_policy(enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask)
        #t2 = time.time()
        #print('I took {} seconds to run'.format(t2-t1))
        return logp
    
class QNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(QNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.action_embedding = nn.Linear(embedding_dim*3, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim, 1)   #output 1 value

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        embedding_feature = self.initial_embedding(node_inputs)
        embedding_feature = self.encoder(input=embedding_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return embedding_feature

    def output_q_values(self, enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask):
        k_size = edge_inputs.size()[2]
        current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, k_size))
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]

        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))

        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))

        enhanced_current_node_feature, attention_weights = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        action_features = torch.cat((enhanced_current_node_feature.repeat(1, k_size, 1), current_node_feature.repeat(1, k_size, 1), neigboring_feature), dim=-1)
        action_features = self.action_embedding(action_features)
        q_values = self.q_values_layer(action_features)

        if edge_padding_mask is not None:
            current_mask = torch.gather(edge_padding_mask, 1, current_index.repeat(1, 1, k_size)).to(
                enhanced_node_feature.device)
        else:
            current_mask = None
        current_mask[:, :, 0] = 1  # don't stay at current position
        #assert 0 in current_mask
        current_mask = current_mask.permute(0, 2, 1)
        zero = torch.zeros_like(q_values).to(q_values.device)
        q_values = torch.where(current_mask == 1, zero, q_values)

        return q_values, attention_weights

    def forward(self, node_inputs, edge_inputs, current_index, node_padding_mask=None, edge_padding_mask=None,
                edge_mask=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        q_values, attention_weights = self.output_q_values(enhanced_node_feature, edge_inputs, current_index, edge_padding_mask, node_padding_mask)
        return q_values, attention_weights
