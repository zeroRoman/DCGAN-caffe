#-*- coding: UTF-8 -*-   #support chinese
#-------------------------------------------------------------
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    #nout: 卷积核（filter)的个数
    #ks:  卷积核的大小
    #stride: 卷积核的步长，默认为1
    #pad: 扩充边缘，默认为0，不扩充。 扩充的时候是左右、上下对称的
    #lr_mult: 学习率的系数，最终的学习率是这个数乘以solver.prototxt配置文件中的base_lr。
    #如果有两个lr_mult, 则第一个表示权值的学习率，第二个表示偏置项的学习率。
    #一般偏置项的学习率是权值学习率的两倍
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'))
    return conv, L.PReLU(conv, in_place=True)#in_place为同址运算

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split,path_data):
    n = caffe.NetSpec()#获取Caffe的一个Net，后续不断的填充net, mean=(104.00699, 116.66877, 122.67892)
    pydata_params = dict(split=split, mean=(91.62,103.25,108.89), seed=1337)#mean=(91.1771497155661,99.7651699986503,90.5919853841616)
    if split == 'train':
        pydata_params['sbdd_dir'] = path_data#'../../data/sbdd/dataset'
        pylayer = 'SBDDSegDataLayer'
    else:
        pydata_params['voc_dir'] = path_data#'../../data/pascal/VOC2011'
        pylayer = 'VOCSegDataLayer'
    
    #数据输入层
    n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
    
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
    
    n.g_concat = crop(n.g_conv7_2, n.data)
    n.concat = L.Concat(n.data,n.g_concat,concat_param=dict(axis=1))

    n.g_conv8_1, n.g_relu8_1  =  conv_relu(n.concat, 32)
    n.g_bn_8 =  L.BatchNorm(n.g_conv8_1,param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)])
    n.g_scale_8 = L.Scale(n.g_bn_8,param=dict(decay_mult=1,lr_mult=0),filler=dict(type="constant",value=0))
    n.g_conv8_2 = L.PReLU(n.g_scale_8, in_place=True)   
    
    n.g_conv9_1, n.g_relu9_1 =  conv_relu(n.g_conv8_2, 2)
    
    n.concat_2 = L.Concat(n.data,n.g_conv9_1)
  
    n.d_conv1_1, n.d_relu1_1  =  conv_relu(n.concat_2, 64)
    n.d_drop_1 = L.Dropout(n.d_conv1_1, dropout_ratio=0.25, in_place=True)
    n.d_pool_1 = max_pool(n.d_drop_1) #ave
    
    
    n.d_conv2_1, n.d_relu2_1  =  conv_relu(n.d_pool_1, 128)
    n.d_drop_2 = L.Dropout(n.d_conv2_1, dropout_ratio=0.25, in_place=True)

    n.d_conv3_1, n.d_relu3_1  =  conv_relu(n.d_drop_2, 256)
    n.d_drop_3 = L.Dropout(n.d_conv3_1, dropout_ratio=0.25, in_place=True)
    n.d_pool_3 = max_pool(n.d_drop_3) 
    
    n.d_conv4_1 = L.Deconvolution(n.d_pool_3,
        convolution_param=dict(num_output=128, kernel_size=2, stride=2, bias_term=False),
        param=[dict(lr_mult=0)])        

    n.d_prelu_4 = L.PReLU(n.d_conv4_1, in_place=True)   
    n.d_drop_4 = L.Dropout(n.d_prelu_4, dropout_ratio=0.5, in_place=True)
    
    n.d_conv5_1 = L.Deconvolution(n.d_drop_4,
        convolution_param=dict(num_output=2, kernel_size=2, stride=2, bias_term=False),
        param=[dict(lr_mult=0)])        

    #n.d_fc_5 = L.InnerProduct(n.d_drop_4, num_output=2, 
      #                     weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'),
       #                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    
    n.score = crop(n.data, n.d_conv5_1)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()
 
def make_net(Path_TrainProtoTxt,Path_ValProtoTxt,Path_DeployTxt,Path_TestTxt,\
             Path_TrainData,Path_ValData,Path_TestData):
    with open(Path_TrainProtoTxt, 'w') as f:
        f.write(str(fcn('train',Path_TrainData)))

    with open(Path_ValProtoTxt, 'w') as f:
        f.write(str(fcn('seg11valid',Path_ValData)))
        
    with open(Path_DeployTxt, 'w') as f:
        f.write(str(fcn('seg11valid',Path_TestData)))   
        
#     with open(Path_TestTxt, 'w') as f:
#         f.write(str(fcn('seg11valid',Path_TestData)))
    
if __name__ == '__main__':
    make_net()