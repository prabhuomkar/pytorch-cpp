import torch
import torchvision
import argparse

class EncoderCNNBackbone(torch.nn.Module):
    def __init__(self):
        super(EncoderCNNBackbone, self).__init__()
        
        resnet_children = list(torchvision.models.resnet50(pretrained=True).children())[:-2]
        self.layers = torch.nn.Sequential(*resnet_children)
        out_features = self.layers[-1][-1].conv3.out_channels
            
        self.register_buffer("out_channels", torch.tensor(out_features))
        
    def forward(self, X):
        return self.layers(X)
    
def create_backbone_torchscript_module():
    backbone = EncoderCNNBackbone()
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    backbone.eval()
    
    example_size = 224
    
    example = torch.rand(1, 3, example_size, example_size)
    
    return torch.jit.trace(backbone, example)

def main():
    backbone_script_module = create_backbone_torchscript_module()

    # Serialize scriptmodule to a file.
    filename = "encoder_cnn_backbone.pt"
    backbone_script_module.save(filename)
    print(f"Successfully created scriptmodule file {filename}.")


if __name__ == "__main__":
    main()
