import matplotlib.pyplot as plt
import torchvision

def show_images(lr, sr):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(lr.permute(1,2,0))
    axs[0].set_title("Low-Res")
    axs[1].imshow(sr.permute(1,2,0))
    axs[1].set_title("Super-Resolved")
    plt.show()
