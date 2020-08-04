# import metrics
import os
import math
import random
import time
from PIL import Image

import torch # pytorch의 tensor와 그와 관련된 기본 연산 등을 지원
import torch.nn as nn # 여러 딥러닝 layer와 loss, 함수 등을 클래스 형태로 지원
import torch.nn.functional as F # 여러 loss, 함수 등을 function 형태로 지원
import torch.optim as optim # 여러 optimizer를 지원

from torchvision.datasets import ImageFolder # (img, label) 형태의 데이터셋 구성을 쉽게 할 수 있도록 지원
import torchvision.transforms as T # 이미지 전처리를 지원
import torchvision.utils # 여러가지 편리한 기능을 지원 (ex. grid 이미지 만들기 등)
import torchvision.models as models # VGG, ResNet 등을 바로 로드할 수 있도록 지원
from torch.utils.data import DataLoader # 데이터 로더를 쉽게 만들 수 있도록 지원

from torch.utils.tensorboard import SummaryWriter # Tensorflow의 Tensorboard를 지원

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def cos_dist(x1, x2):
    '''
    목적 : 벡터 x1, x2 사이의 Cosine similary를 계산
    
    인자 :
    x1, x2 : 두 벡터
    '''
    return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))

def fixed_img_list(lfw_pair_text, test_num):
    '''
    목적 : 중간 테스트 때 계속 사용될 고정 test 이미지 리스트 생성
    
    인자:
    lfw_pair_text : pair가 만들어진 이미지 경로
    test_num : 테스트에 사용할 pair 개수
    '''
    f = open(lfw_pair_text, 'r')
    lines = []

    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    random.shuffle(lines)
    lines = lines[:test_num]
    return lines

def verification(net, pair_list, tst_data_dir, img_size, gray_scale=True):
    '''
    목적 : Face verification 테스트 수행
    
    인자:
    net : 네트워크
    pair_list : pair가 만들어진 이미지 경로의 리스트
    tst_data_dir : 테스트 이미지가 있는 루트 디렉토리
    img_size : 테스트에 사용할 이미지 사이즈
    gray_scale : 흑백으로 변환 여부
    '''
    
    '''
    STEP 1 : 주어진 이미지 pair에서 feature 를 뽑아 similarity, label 리스트 생성
    '''
    similarities = []
    labels = []

    # 이미지 전처리
    trans_list = []
    
    if gray_scale:
        trans_list += [T.Grayscale(num_output_channels=1)]
        
    trans_list += [T.CenterCrop((178, 178)),
                   T.Resize((img_size, img_size)),
                   T.ToTensor()]
    
    if gray_scale:
        trans_list += [T.Normalize(mean=(0.5,), std=(0.5,))]
    else:
        trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = T.Compose(trans_list)

    # 주어진 모든 이미지 pair에 대해 similarity 계산
    net.eval()
    with torch.no_grad(): # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(pair_list):
            # Read paired images
            path_1, path_2, label = pair.split(' ')
            img_1 = t(Image.open(os.path.join(tst_data_dir, path_1))).unsqueeze(dim=0).to(dev)
            img_2 = t(Image.open(os.path.join(tst_data_dir, path_2))).unsqueeze(dim=0).to(dev)
            imgs = torch.cat((img_1, img_2), dim=0)

            # Extract feature and save
            features = net.extract_feature(imgs).cpu()
            similarities.append(cos_dist(features[0], features[1]))
            labels.append(int(label))
            
    '''
    STEP 2 : similarity와 label로 verification accuracy 측정
    '''
    best_accr = 0.0
    best_th = 0.0

    # 각 similarity들이 threshold의 후보가 된다
    list_th = similarities
    
    # list -> tensor
    similarities = torch.stack(similarities, dim=0)
    labels = torch.ByteTensor(labels)

    # 각 threshold 후보에 대해 best accuracy를 측정
    for i, th in enumerate(list_th):
        pred = (similarities >= th)
        correct = (pred == labels)
        accr = torch.sum(correct).item() / correct.size(0)

        if accr > best_accr:
            best_accr = accr
            best_th = th.item()

    return best_accr, best_th


class ArcMarginProduct(nn.Module):
    '''
    목적 : Arc marin 을 포함한 last fc layer의 구현
    
    인자 :
    in_features : feature의 dimension = 512
    out_features : class 개수 = 10535
    '''
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # fc의 parameter 만들기
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        '''
        Step 1. cos(theta + m) 계산하기
        '''
        # cosine : cos(theta)
        # [N, in_features] * [in_features, out_features] = [N, out_features] / normalize-> cos dist == cos
        # linear(x, W) = x * W^T = [N, in_features] * [in_features, out_features] = [N, out_features]
        # F.normalize는 디폴트가 dim = 1 (줄이고자하는 dimension)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # c^2 + s^2 = 1 
        sine = torch.sqrt((1.00000001 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        '''
        Step 2. cos(theta + m) 에서 dim=1에 대해 y_i에 해당하는 부분만 남기고 나머지는 cos(theta)로 되돌리기 
        '''
        one_hot = torch.zeros(cosine.size()).to(dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        '''
        Step 3. 이 output이 softmax -> cross entropy loss로 흘러가면 된다.
        '''
        return output

class FeatureNet_50(nn.Module):
    '''
    목적 : ResNet-50을 이용한 backbone network(feature extractor) 구현 
    
    인자 :
    feature_dim : feature의 dimension
    gray_scale : 이미지를 gray scale로 받았는지 여부
    '''
    def __init__(self, feature_dim, gray_scale=True):
        super(FeatureNet_50, self).__init__()
        # Pytorch에서 이미 구현되어 있는 resnet-50 불러오기
        resnet = models.resnet50(pretrained=False)
        
        # 이런식으로 불러온 resnet을 조건에 맞게 변경할 수 있다.
        if gray_scale:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # resnet의 마지막 conv block까지만 남기고 나머지 부분 잘라내기
        self.backbone = nn.Sequential(* list(resnet.children())[0:-2])
        
        # resnet의 마지막 conv block 뒤쪽으로 새로 붙을 layer 들
        self.bn_4 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048 * 4 * 4, feature_dim)
        self.bn_5 = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.bn_4(out)
        out = self.dropout(out)
        
        # FC layer를 지나기 전에는 reshape 과정이 필요하다.
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.bn_5(out)
        return out

class ArcFaceNet(nn.Module):
    '''
    ArcMarginProduct와 FeatureNet-50 을 결합한 ArcFace 모델의 구현
    '''
    def __init__(self, feature_dim, cls_num, gray_scale=True):
        super(ArcFaceNet, self).__init__()
        self.feature_net = FeatureNet_50(feature_dim, gray_scale=gray_scale)
        self.classifier = ArcMarginProduct(feature_dim, cls_num)

    # 끝까지 Forward 하여 logit을 return
    def forward(self, x, label):
        out = self.feature_net(x)
        out = self.classifier(out, label)
        return out
    
    # Feature만 return
    def extract_feature(self, x):
        out = self.feature_net(x)
        return out

# 경로
tst_data_dir = '../data/lfw'
lfw_pair_text = 'lfw_test_part.txt'
weight_dir = 'weight/arcface'

# 이미지 세팅
resize = 128
gray_scale = True

# 모델 세팅
feature_dim=512
cls_num=10575

# 테스트 세팅
test_num=6000

# 모델 로드 및 테스트 준비
net = ArcFaceNet(feature_dim=feature_dim, 
                 cls_num=cls_num, 
                 gray_scale=gray_scale).to(dev)

net.load_state_dict(torch.load('weight/arcface_pretrained/ckpt_280000.pkl'))
net.eval()

# 테스트
fixed_test_pair = fixed_img_list(lfw_pair_text, test_num)
accr, th = verification(net, fixed_test_pair, tst_data_dir, resize, gray_scale)
print(accr, th)