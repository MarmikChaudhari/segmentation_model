from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch 
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("..")


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image = cv2.imread('/Users/marmik/Downloads/vegtable.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image)

img_org = cv2.imread('/Users/marmik/Downloads/vegtable.jpg')
img_org = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



#original image 
#plt.figure(figsize=(10,10))
#plt.imshow(image)
#plt.axis('on')
#plt.show()

sam = sam_model_registry["vit_b"](checkpoint="/Users/marmik/Downloads/sam_vit_b_01ec64.pth")
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(sam)


masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

# Create a directory to save the segmented images
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the segmented parts of the image as individual images
for i, ann in enumerate(masks):
    mask = ann['segmentation']
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    roi = image[y:y+h, x:x+w]

    #mask = np.zeros_like(image)
    #mask[y:y+h, x:x+w] = roi

    #segmented_image = np.zeros_like(image)
    #segmented_image[mask ==1] = image[mask==1]

    #remove the mask from the segmented image
    #mask_resized = cv2.resize(mask.astype(np.uint8), (roi.shape[1], roi.shape[0]))
    #roi[mask_resized.astype(bool)] = [0,0,0]
    original_roi = img_org[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_dir, f'segmented_{i}.jpg'), original_roi)


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


