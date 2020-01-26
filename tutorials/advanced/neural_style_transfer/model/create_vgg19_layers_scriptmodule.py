import torch
import torchvision


def main():
    # Download and load the pretrained VGG19 layers.
    vgg_19_layers = torchvision.models.vgg19(pretrained=True).features

    for param in vgg_19_layers.parameters():
        param.requires_grad = False

    # Serialize scriptmodule to a file.
    filename = "vgg19_layers.pt"
    vgg_19_layers.save(filename)
    print(f"Successfully created scriptmodule file {filename}.")


if __name__ == "__main__":
    main()
