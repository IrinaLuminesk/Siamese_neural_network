from torch.nn import functional as F
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights,\
    densenet201, DenseNet201_Weights,\
    vgg16, VGG16_Weights, \
    mobilenet_v2, MobileNet_V2_Weights, \
    efficientnet_b4, EfficientNet_B4_Weights, \
    inception_v3, Inception_V3_Weights
import torch

class Model(nn.Module):
    def __init__(self, model_type, pair_type, embedding_dim = 512):
        super().__init__()
        self.model_type = model_type
        self.pair_type = pair_type
        self.embedding_dim = embedding_dim
        self.backbone, self.embedding_layer = self.build_model() 
    def build_model(self):
        match self.model_type:
            case 1: #Resnet50
                resnet_weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=resnet_weights)


                in_features = model.fc.in_features #2048


                backbone = nn.Sequential(*list(model.children())[:-1])

                embedding_layer = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, self.embedding_dim)
                )
                print("Training on Resnet50 architecture")
                return backbone, embedding_layer
            # case 2: #VGG16
            #     vgg16_weights = VGG16_Weights.DEFAULT
            #     model = vgg16(weights=vgg16_weights)

            #     # vgg16_classifier = list(model.classifier.children())[:6]
            #     in_features = model.classifier[0].in_features #25088

            #     backbone = nn.Sequential(*list(model.children())[:-1])
            #     embedding_layer = nn.Sequential(
            #         nn.Linear(in_features, self.embedding_dim),
            #         nn.BatchNorm1d(self.embedding_dim)
            #     )
            #     print("Training on VGG16 architecture")
            #     return backbone, embedding_layer
            # case 3: #Xception
            #     model = timm.create_model(
            #         'xception65',
            #         pretrained=True
            #     )

            #     in_features = model.get_classifier().in_features
                
            #     backbone = nn.Sequential(*list(model.children())[:-1])
            #     embedding_layer = nn.Sequential(
            #         nn.Linear(in_features, self.embedding_dim),
            #         nn.BatchNorm1d(self.embedding_dim)
            #     )
            #     print("Training on Xception65 architecture")
            #     return backbone, embedding_layer
            # case 4: #EfficientNetB4
            #     model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

            #     in_features = model.classifier[1].in_features #1792

            #     backbone = nn.Sequential(*list(model.children())[:-1])
            #     embedding_layer = nn.Sequential(
            #         nn.Linear(in_features, self.embedding_dim),
            #         nn.BatchNorm1d(self.embedding_dim)
            #     )
            #     print("Training on EfficientNetB4 architecture")
            #     return backbone, embedding_layer
            # case 5: #DenseNet201
            #     densenet_Weights = DenseNet201_Weights.DEFAULT
            #     model = densenet201(weights=densenet_Weights)

            #     in_features = model.classifier.in_features #1920
                
            #     backbone = nn.Sequential(*list(model.children())[:-1])
            #     embedding_layer = nn.Sequential(
            #         nn.Linear(in_features, self.embedding_dim),
            #         nn.BatchNorm1d(self.embedding_dim)
            #     )
            #     print("Training on DenseNet201 architecture")
            #     return backbone, embedding_layer
            # case 6: #MobileNet
            #     mobilenetv2_weights = MobileNet_V2_Weights.DEFAULT
            #     model = mobilenet_v2(weights=mobilenetv2_weights)

            #     in_features = model.classifier[1].in_features #1280

            #     backbone = nn.Sequential(*list(model.children())[:-1])
            #     embedding_layer = nn.Sequential(
            #         nn.Linear(in_features, self.embedding_dim),
            #         nn.BatchNorm1d(self.embedding_dim)
            #     )
            #     print("Training on MobileNetV2 architecture")
            #     return backbone, embedding_layer
            # case 7: #Inception-v3
            #     inception_v3_weight = Inception_V3_Weights.DEFAULT
            #     model = inception_v3(weights=inception_v3_weight)

            #     in_features = model.fc.in_features #2048
                
            #     backbone = nn.Sequential(*list(model.children())[:-2])
            #     embedding_layer = nn.Sequential(
            #         nn.Linear(in_features, self.embedding_dim),
            #         nn.BatchNorm1d(self.embedding_dim)
            #     )
            #     print("Training on Inception V3 architecture")
            #     return backbone, embedding_layer
            case _:
                raise ValueError("Not valid model type")
    def forward_once(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding_layer(x)

        # L2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x
    
    def forward(self, x1, x2, x3=None):
        match self.pair_type:
            case 1:
                x1 = self.forward_once(x1)
                x2 = self.forward_once(x2)
                return x1, x2
            case 2:
                x1 = self.forward_once(x1)
                x2 = self.forward_once(x2)
                x3 = self.forward_once(x3)
                return x1, x2, x3