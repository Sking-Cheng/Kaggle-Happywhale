# Kaggle-Happywhale
Happywhale - Whale and Dolphin Identification Silver🥈 Solution (26/1588)
## 竞赛方案思路

1. 图像数据预处理-标志性特征图片裁剪：首先根据开源的标注数据训练**YOLOv5x6**目标检测模型，将训练集与测试集数据裁剪出背鳍或者身体部分;
2. 背鳍图片特征提取模型：将训练集数据划分为训练与验证两部分，训练 **EfficientNet_B6 / EfficientNet_V2_L / NFNet_L2 **（backone）三个模型，并且都加上了**GeM Pooling 和 Arcface 损失函数**，有效增强类内紧凑度和类间分离度;
3. 聚类与排序：利用最终训练完成的backone模型分别提取训练集与测试集的嵌入特征，所有模型都会输出一个**512维的Embedding**，将这些特征 concatenated 后获得了一个 **512×9=4608 **维的特征向量，将训练集的嵌入特征融合后训练**KNN**模型，然后推断测试集嵌入特征距离，排序获取top5类别，作为预测结果，最后使用new_individual替换进行后处理，得到了top2%的成绩。

#### Model

```python
class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained) 

        if 'efficientnet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
        elif 'nfnet' in model_name:
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
            self.model.head.global_pool = nn.Identity()

        self.pooling = GeM() 
        self.embedding = nn.Sequential(
                            nn.BatchNorm1d(in_features),
                            nn.Linear(in_features, embedding_size)
                            )
        # arcface
        self.fc = ArcMarginProduct(embedding_size,
                                   CONFIG["num_classes"], 
                                   s=CONFIG["s"],
                                   m=CONFIG["m"], 
                                   easy_margin=CONFIG["easy_margin"], 
                                   ls_eps=CONFIG["ls_eps"]) 

    def forward(self, images, labels):
        features = self.model(images)  
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features) # embedding
        output = self.fc(embedding, labels) # arcface
        return output
    
    def extract(self, images):
        features = self.model(images) 
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features) # embedding
        return embedding
```



#### ArcFace

```python
# Arcface
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.s = s
        self.m = m 
        self.ls_eps = ls_eps 
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m) # cos margin
        self.sin_m = math.sin(m) # sin margin
        self.threshold = math.cos(math.pi - m) # cos(pi - m) = -cos(m)
        self.mm = math.sin(math.pi - m) * m # sin(pi - m)*m = sin(m)*m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) 
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)) 
        phi = cosine * self.cos_m - sine * self.sin_m # cosθ*cosm – sinθ*sinm = cos(θ + m)
        phi = phi.float() # phi to float
        cosine = cosine.float() # cosine to float
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # if cos(θ) > cos(pi - m) means θ + m < math.pi, so phi = cos(θ + m);
            # else means θ + m >= math.pi, we use Talyer extension to approximate the cos(θ + m).
            # if fact, cos(θ + m) = cos(θ) - m * sin(θ) >= cos(θ) - m * sin(math.pi - m)
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
            
        # https://github.com/ronghuaiyang/arcface-pytorch/issues/48
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=CONFIG['device'])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # label smoothing
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) #验证是否匹配正确 
        output *= self.s

        return output
```



## 冲榜历程

1. 使用Yolov5切分 fullbody数据 和 backfins数据；
2. 使用小模型tf_efficientnet_b0_ns + ArcFace 作为 Baseline，训练fullbody 512size, 使用kNN 搜寻，搭建初步的pipline，Public LB : 0.729；
3. 加入new_individual后处理，Public LB : 742；
4. 使用fullbody 768size图像，并调整了数据增强， Public LB : 0.770；
5. 训练 tf_efficientnet_b6_ns ，以及上述所有功能微调，Public LB：0.832；
6. 训练 tf_efficientnetv2_l_in21k，以及上述所有功能微调，Public LB：0.843；
7. 训练 eca_nfnet_l2，以及上述所有功能微调，Public LB：0.854；
8. 将上述三个模型的5Fold，挑选cv高的，进行融合，Public LB：0.858；



## 代码、数据集

+ 代码

  + Happywhale_crop_image.ipynb  # 裁切fullbody数据和backfin数据
  + Happywhale_train.ipynb  # 训练代码 (最低要求GPU显存不小于12G)
  + Happywhale_infernce.ipynb # 推理代码以及kNN计算和后处理
+ 数据集

  + [官方数据集](https://www.kaggle.com/competitions/happy-whale-and-dolphin/data)
  + datasets文件夹
