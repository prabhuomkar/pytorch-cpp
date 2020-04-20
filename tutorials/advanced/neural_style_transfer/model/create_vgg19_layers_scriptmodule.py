import torch
import torchvision


def main():
    # Download and load the pretrained VGG19 layers.
    vgg_19_layers = torchvision.models.vgg19(pretrained=True).features

    for param in vgg_19_layers.parameters():
        param.requires_grad = False

    example = torch.rand(1, 3, 224, 224)

    traced_script_module = torch.jit.trace(vgg_19_layers, example)

    # Serialize scriptmodule to a file.
    filename = "vgg19_layers.pt"
    traced_script_module.save(filename)
    print(f"Successfully created scriptmodule file {filename}.")


if __name__ == "__main__":
    main()
