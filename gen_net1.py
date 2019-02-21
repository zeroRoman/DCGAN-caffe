#-*- coding: UTF-8 -*-   #support chinese
# Path_Caffe = '/root/caffe-master/python'
# import sys
# sys.path.append(Path_Caffe)     #修改系统路径
import caffe
from PIL import Image 
import Image  
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'))
    #return conv, L.PReLU(conv, in_place=True)#in_place为同址运算
    return conv, L.PReLU(conv, in_place=True)#in_place为同址运算

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def netStructure(ForFunc, dataType, Path_InData, batch_size, ImgSize):
    n = caffe.NetSpec()
    
    if ForFunc == 'Train'  or  ForFunc == 'Test' :
        if dataType == 'lmdb':
            n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=Path_InData,
                                     transform_param=dict(scale=1./255), ntop=2) #ntop表示blob的个数
        else:
            n.data, n.label = L.HDF5Data(batch_size=batch_size, source=Path_InData, ntop=2)

    if ForFunc == 'Deploy':
        n.data = L.Input(shape=ImgSize)
                
    # 卷积层和池化层, conv_relu(bottom, nout, ks=3, stride=1, pad=1)
    n.g_conv1_1, n.g_relu1_1  =  conv_relu(n.data, 16)
    n.g_bn_1 =  L.BatchNorm(n.g_conv1_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_1 = L.Scale(n.g_bn_1,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv1_2 = L.PReLU(n.g_scale_1, in_place=True)   
    
    n.g_conv2_1, n.g_relu2_1  =  conv_relu(n.g_conv1_2, 32)
    n.g_bn_2 =  L.BatchNorm(n.g_conv2_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_2 = L.Scale(n.g_bn_2,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv2_2 = L.PReLU(n.g_scale_2, in_place=True)   
    n.g_pool_2 = max_pool(n.g_conv2_2)
    
    
    n.g_conv3_1, n.g_relu3_1  =  conv_relu(n.g_pool_2, 64)
    n.g_bn_3 =  L.BatchNorm(n.g_conv3_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_3 = L.Scale(n.g_bn_3,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv3_2 = L.PReLU(n.g_scale_3, in_place=True)   
    n.g_pool_3 = max_pool(n.g_conv3_2)    
    
    n.g_conv4_1, n.g_relu4_1  =  conv_relu(n.g_pool_3, 128)
    n.g_bn_4 =  L.BatchNorm(n.g_conv4_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_4 = L.Scale(n.g_bn_4,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv4_2 = L.PReLU(n.g_scale_4, in_place=True)   
   # n.g_pool_4 = max_pool(n.g_conv4_2)    
   
    n.g_conv5_1, n.g_relu5_1  =  conv_relu(n.g_conv4_2, 256)
    n.g_bn_5 =  L.BatchNorm(n.g_conv5_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_5 = L.Scale(n.g_bn_5,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv5_2 = L.PReLU(n.g_scale_5, in_place=True)   


    n.g_conv6_1 = L.Deconvolution(n.g_conv5_2,
                                    convolution_param=dict(num_output=128, kernel_size=2, stride=2, bias_term=False),
                                      param=[dict(lr_mult=0)])    
    n.g_bn_6 =  L.BatchNorm(n.g_conv6_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_6 = L.Scale(n.g_bn_6,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv6_2 = L.PReLU(n.g_scale_6, in_place=True)   


    n.g_conv7_1 = L.Deconvolution(n.g_conv6_2,
        convolution_param=dict(num_output=64, kernel_size=2, stride=2, bias_term=False),
        param=[dict(lr_mult=0)])    
    n.g_bn_7 =  L.BatchNorm(n.g_conv7_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_7 = L.Scale(n.g_bn_7,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv7_2 = L.PReLU(n.g_scale_7, in_place=True)   
    
    n.g_concat = crop(n.g_conv1_2, n.g_conv7_2)
    n.concat = L.Concat(n.g_conv1_2,n.g_concat, concat_param=dict(axis=1) )

    n.g_conv8_1, n.g_relu8_1  =  conv_relu(n.concat, 32)
    n.g_bn_8 =  L.BatchNorm(n.g_conv8_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_8 = L.Scale(n.g_bn_8,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv8_2 = L.PReLU(n.g_scale_8, in_place=True)   
    
    n.g_conv9_1, n.g_relu9_1 =  conv_relu(n.g_conv8_2, 2)
    
    

#    n.g_bn_8 =  L.BatchNorm(n.g_conv8_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
  #  n.g_scale_8 = L.Scale(n.g_bn_8,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
  #  n.g_conv8_2 = L.PReLU(n.g_scale_8, in_place=True)   
    
    n.score_fc = L.InnerProduct(n.g_relu9_1, num_output=2, 
                           weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'),
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
   
    if ForFunc == 'Train':
        n.accuracy = L.Accuracy(n.score_fc, n.label)
        n.loss =  L.SoftmaxWithLoss(n.score_fc, n.label)
    if ForFunc == 'Test':
        n.accuracy = L.Accuracy(n.score_fc, n.label)


    return n.to_proto()

#produce and save prototxt
def make_net(dataType, path_TrainPrototxt, path_ValPrototxt, Path_TestPrototxt,Path_DeployPrototxt,\
             path_TrainData, path_ValData, batchSize_Train, batchSize_Val, TrainImgSize,
             path_TestData, path_DeployData, batchSize_Test, TestImgSize):
   
    #for train
    with open(path_TrainPrototxt, 'w') as f:
        f.write(str(netStructure('Train', dataType, path_TrainData, batchSize_Train,TrainImgSize)))

    #for val
    with open(path_ValPrototxt, 'w') as f:
        f.write(str(netStructure('Train', dataType, path_ValData, batchSize_Val,TrainImgSize)))

    #for test
    with open(Path_TestPrototxt, 'w') as f:
        f.write(str(netStructure('Test', dataType, path_TestData, batchSize_Test,TestImgSize)))
    with open(Path_DeployPrototxt, 'w') as f:
        f.write(str(netStructure('Deploy', dataType, path_DeployData, batchSize_Test,TestImgSize)))

if __name__ == '__main__':
    make_net()