import torch
import torchvision


def main():
    # Source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
    # Download and load the pretrained ResNet-18.
    model = torchvision.models.resnet18(pretrained=True)

    # If you want to finetune only the top layer of the model, set as below
    for param in model.parameters():
        param.requires_grad = False

    # Replace the top layer for finetuning.
    model.fc = torch.nn.Linear(model.fc.in_features, 100)

    # Source: https://pytorch.org/tutorials/advanced/cpp_export.html#converting-to-torch-script-via-tracing
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Serialize scriptmodule to a file.
    filename = "resnet18_scriptmodule.pt"
    traced_script_module.save(filename)
    print(f"Successfully created scriptmodule file {filename}.")


if __name__ == "__main__":
    main()
