from pytorch_segnet import *
import numpy as np
import torch.utils.data as Data
from PIL import Image
from collections import  Counter
import sys
import csv


device = torch.device("cuda " if torch.cuda.is_available() else "cpu")

f_score=np.zeros((19,1))

label = Image.open('Datasets/dataset/trainData/labelpng/aachen_000000_000019_gtFine_labelTrainIds.png')
label = np.array(label)
#label = label.flatten()

def fgsm_attack(image, epsilon, data_grad):
    # input the original image
    sign_data_grad = data_grad.sign()
    # generate the perturbation
    perturbed_image = image + epsilon*sign_data_grad
    # make the pixel point range in 0 to 1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)# make the perturbation range in 0 to 1
    # return the perturbation image
    return perturbed_image

def adver_test(net, device, test_loader, epsilon):

    adv_examples = []  # for save the adversarial attack examples

    # extract the image from dataset
    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        # set the tensor property
        data.requires_grad = True

        # forward propagation

        output1, relu_value, output = net(data)


        # loss function
        loss = F.nll_loss(output, target.long(),ignore_index=255).cpu()

        # reset gradients
        net.zero_grad()

        # back propagation
        loss.backward()

        # get the gradients
        data_grad = data.grad.data

        # apply the FGSM function
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        adv_examples.append(perturbed_data)

    # return the adversarial attacks example
    return  adv_examples



def test(SegNet):

    SegNet.load_state_dict(torch.load(WEIGHTS,map_location=torch.device('cpu')))
    SegNet.eval()

    paths = os.listdir(SAMPLES)

    adv_examples = adver_test(SegNet, device, train_loader, epsilons[1])
    adv_examples = adv_examples[0]

    for path in paths:

        image_src = cv.imread(SAMPLES + path)
        image = cv.resize(image_src, (416, 416))

        image = image / 255.0
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, dim=0)


        output, relu_value, output1 = SegNet(image)
        output_ad, relu_value_ad, output1_ad = SegNet(adv_examples)

        output = torch.squeeze(output)
        output_ad = torch.squeeze(output_ad)

        output = output.argmax(dim=0)
        output_ad = output_ad.argmax(dim=0)

        output_np = cv.resize(np.uint8(output), (2048, 1024))
        adver_output = cv.resize(np.uint8(output_ad), (2048, 1024))

        # label = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky'
        #          ,'person','rider','car','truck','bus','train','motocycle','bicycle']

        colors = COLORS



        return output_np,relu_value, adver_output,relu_value_ad


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=19, help="class")
parser.add_argument("--weights", type=str, default="weights/SegNet_weights1625783040.5590682.pth", help="path")
parser.add_argument("--colors", type=int, default=[[0, 255, 0], [255, 0, 0],[0, 0, 255],[111, 74, 0],[70, 70, 70],[128, 64, 128],[0, 0, 0],[102, 102, 156],[190, 153, 153],[150, 100, 100],[107, 142, 35],
                 [152, 251, 152],[70, 130, 180],[220, 220, 0],[119, 11, 32],[215, 166, 66],[66, 88, 99],[154, 25, 244],[10, 155, 83]], help="mask")
parser.add_argument("--samples", type=str, default="samples//", help="test path")
parser.add_argument("--outputs", type=str, default="outputs//", help="save path")
parser.add_argument("--outputs1", type=str, default="outputs1//", help="save_ad path")
parser.add_argument("--train_txt", type=str, default="train.txt", help="label")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
opt = parser.parse_args()

#print(opt)

CLASS_NUM = opt.class_num
WEIGHTS = opt.weights
COLORS = opt.colors
SAMPLES = opt.samples
OUTPUTS = opt.outputs
OUTPUTS1 = opt.outputs1
TXT_PATH = opt.train_txt
BATCH_SIZE = opt.batch_size

train_data = MyDataset(txt_path=TXT_PATH)
epsilons = [0, .05, .1, .15, .2, .25, .3]
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

SegNet = SegNet(3, CLASS_NUM)
prediction_output, output_relu, ad_prediction_output, output_relu_ad = test(SegNet)

prediction_output = np.array(prediction_output)
#prediction_output = prediction_output.flatten()
ad_prediction_output = np.array(ad_prediction_output)
ad_prediction_output = ad_prediction_output.flatten()

output_relu = output_relu.tolist()
output_relu = np.array(output_relu)
output_relu = output_relu.flatten()


output_relu_ad = output_relu_ad.tolist()
output_relu_ad = np.array(output_relu_ad)
output_relu_ad = output_relu_ad.flatten()
temp_relu_ad = output_relu_ad

if len(output_relu) == len(output_relu_ad):

    for i in range(len(output_relu)):
        if output_relu[i] > 0:
            output_relu[i] = 1
        else:
            output_relu[i] = 0

    for j in range(len(temp_relu_ad)):
        if temp_relu_ad[j] > 0:
            temp_relu_ad[j] = 1
        else:
            temp_relu_ad[j] = 0

class Hammingdistance():
    def H_d(self,x,y):
        hammingdistance = 0
        for i in range(len(x)):
            if x[i] != y[i]:
                hammingdistance += 1
        return  hammingdistance

Hammingdistance_value = Hammingdistance()
Hammingdistance_value = Hammingdistance_value.H_d(output_relu, temp_relu_ad)

mask = (label >= 0) & (label < 19)
label_1 = 19*label[mask].astype('int') + prediction_output[mask]
count = np.bincount(label_1,minlength=19*19)
confusion_matrix = count.reshape(19,19)


if __name__ == '__main__':
    precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    for i in range(0, 19):
        f_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

    print(precision)
    print(recall)
    print(f_score)
    print(Hammingdistance_value)


