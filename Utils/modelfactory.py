import torch
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)

class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        model_name = model_name.lower()
        device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if model_name == 'deeplabv3plus':
            from VSmethod.SemanticSegment.DEEPLABV3PLUS import DeepLab
            model = DeepLab(
                num_classes=kwargs['classNum'],
            )
        elif model_name == 'unet':
            from VSmethod.SemanticSegment.UNET import UNetWithResnet50Encoder
            model = UNetWithResnet50Encoder (
                n_classes=kwargs['classNum'],
            )
        elif model_name == 'dbbanet':
            from VSmethod.SemanticSegment.DBBANET import DBBANet
            model = DBBANet(
                num_class=kwargs['classNum'],
            )
        elif model_name == 'tbkan':
            from Framework.TBKANET import TBKANet
            model = TBKANet(
                num_classes=kwargs['classNum'],
            )
        elif model_name == 'mccanet':
            from VSmethod.SemanticSegment.MCCANET import MCCA
            model = MCCA(
                num_class=kwargs['classNum'],
            )
        elif model_name == 'cmtfnet':
            from VSmethod.SemanticSegment.CMTFNET import CMTFNet
            model = CMTFNet(
                num_classes=kwargs['classNum'],
            )
        elif model_name == 'unetformer':
            from VSmethod.SemanticSegment.UNETFORMER import UNetFormer
            model = UNetFormer(
                num_classes=kwargs['classNum'], 
            )
        elif model_name == 'ukan':
            from VSmethod.SemanticSegment.UKAN import UKAN
            model = UKAN(
                num_classes=kwargs['classNum'], 
            )
        elif model_name == 'hrnet':
            from VSmethod.SemanticSegment.HRNET import HighResolutionNet
            model = HighResolutionNet(
                num_class=kwargs['classNum'],
            )
        elif model_name == 'reaunet':
            from VSmethod.SemanticSegment.REAUNET import REAUNet
            model = REAUNet(
                num_class=kwargs['classNum'], 
            )
        elif model_name == 'segformer':
            from VSmethod.SemanticSegment.SEGFORMER import SegFormerModel
            model = SegFormerModel(
                num_class=kwargs['classNum'], 
                model_size='b0',
                pretrained=True,
            )
        elif model_name == 'labelcorrect':
            from Framework.TBKANET import TBKANet
            model = TBKANet(
                num_classes=kwargs['classNum'],
            )
        elif model_name == 'ablation':
            from Ablation.SemanticSegment.BASE_MSFB_HEGB_KESB import DBBANetzhao
            model = DBBANetzhao(
                num_classes=kwargs['classNum'],
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
            
        return model.to(device)
    
    @staticmethod
    def create_optimizer(model_name, model, args):
        model_name = model_name.lower()
        if model_name in ['deeplabv3plus', 'unet', 'dbbanet', 'tbkan', 'dbbanet_pan', 'mccanet', 'cmtfnet', 'unetformer', 'ukan', 'hrnet', 'reaunet', 'segformer', 'labelcorrect', 'ablation']:
            return torch.optim.AdamW(
                model.parameters(), 
                weight_decay=args.weight_decay,
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    @staticmethod
    def create_scheduler(model_name, optimizer, max_iters, args):
        model_name = model_name.lower()
        if model_name in ['deeplabv3plus', 'unet', 'dbbanet', 'tbkan', 'dbbanet_pan', 'mccanet', 'cmtfnet', 'unetformer', 'ukan', 'hrnet', 'reaunet', 'segformer', 'ablation']:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.max_epochs,
                eta_min=args.min_lr or 1e-6
            )
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    @staticmethod
    def create_loss(model_name, args, device):
        model_name = model_name.lower()
        
        if model_name in ['deeplabv3plus', 'unet', 'dbbanet', 'tbkan', 'dbbanet_pan', 'mccanet', 'cmtfnet', 'unetformer', 'ukan', 'hrnet', 'segformer', 'ablation']:
            from torch.nn import BCEWithLogitsLoss
            return BCEWithLogitsLoss(
                reduction='mean'
            ).to(device)
        elif model_name == 'reaunet':
            from Utils.loss import BCEDiceLoss
            return BCEDiceLoss().to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
