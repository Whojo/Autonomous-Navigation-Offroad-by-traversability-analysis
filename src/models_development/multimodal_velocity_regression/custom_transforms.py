import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2

from params import PROJECT_PATH


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.

    Inputs : PIL Image
    """
    def __init__(self, prob=0.5):
        
        self.prob = prob

    def __call__(self, img):

        p = np.random.uniform(0, 1)

        if p <= self.prob:

            img_np = np.array(img)
            
            h, w, _ = img_np.shape

            mask = np.ones((h, w), np.float32)

            length = np.random.uniform(0, min(h,w))

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.int32(np.clip(y - length // 2, 0, h))
            y2 = np.int32(np.clip(y + length // 2, 0, h))
            x1 = np.int32(np.clip(x - length // 2, 0, w))
            x2 = np.int32(np.clip(x + length // 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

            img_np[:,:,0] = img_np[:,:,0] * mask
            img_np[:,:,1] = img_np[:,:,1] * mask
            img_np[:,:,2] = img_np[:,:,2] * mask

            img = Image.fromarray(img_np)

        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.prob})"
    

class Shadowcasting(object):
    """
    Randomly mask out one or more patches from an image.

    Inputs : PIL Image
    """
    def __init__(self, prob=0.5):

        self.prob = prob

    def __call__(self, img):
        
        p = np.random.uniform(0, 1)

        if p <= self.prob:

            img_np = np.array(img)
            
            h, w, _ = img_np.shape

            mask = np.zeros((h,w))
            point_up, point_down = np.random.uniform(0, w, 2)
            point_up, point_down = int(point_up), int(point_down)
            top, down = [point_up, 0], [point_down, h]

            left_right = np.random.binomial(1, 0.5)

            if left_right == 1:
                corner_up, corner_down = [0, 0], [0, h]
            else:
                corner_up, corner_down = [w, 0], [w, h]
            
            value = np.random.uniform(50, 255)

            cv2.fillConvexPoly(mask, np.array([top, down, corner_down, corner_up]), value)

            img_np[:,:,0] = np.clip(img_np[:,:,0] - mask, 0, 255)
            img_np[:,:,1] = np.clip(img_np[:,:,1] - mask, 0, 255)
            img_np[:,:,2] = np.clip(img_np[:,:,2] - mask, 0, 255)

            img = Image.fromarray(img_np)

        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.prob})"
    

if __name__ == "__main__":
    # Open the first image of the sample dataset
    image = Image.open(str(PROJECT_PATH / "datasets/dataset_multimodal_siamese_png/images/00000.png"))
    # image = Image.open(str(PROJECT_PATH / "datasets/dataset_sample_bag/zed_node_rgb_image_rect_color/00000.png"))
    
    # Create a new figure
    plt.figure()

    # Display the original image
    plt.subplot(321)
    plt.imshow(image)
    plt.title("Original image")
    
    # Resize the image
    image_resized = transforms.Resize(200)(image)
    
    # Display the resized image
    plt.subplot(322)
    plt.imshow(image_resized)
    plt.title("Resized image")
    
    
    # Normalize the image (in fact tensor) (mean and standard deviation are
    # pre-computed on the ImageNet dataset)
    tensor_normalized = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])(image)
    
    # Convert the tensor to a PIL Image
    image_normalized = transforms.ToPILImage()(tensor_normalized)
    
    # Display the normalized image
    plt.subplot(323)
    plt.imshow(image_normalized)
    plt.title("Normalized image")
    
    tensor_shadow = transforms.Compose([
        Shadowcasting(1),
        transforms.ToTensor()
    ])(image)

    image_shadow = transforms.ToPILImage()(tensor_shadow)
    
    plt.subplot(324)
    plt.imshow(image_shadow)
    plt.title("Shadow image")

    tensor_cutout = transforms.Compose([
        Cutout(1),
        transforms.ToTensor()
    ])(image)

    image_cutout = transforms.ToPILImage()(tensor_cutout)
    
    plt.subplot(325)
    plt.imshow(image_cutout)
    plt.title("Cutout image")
    
    plt.show()