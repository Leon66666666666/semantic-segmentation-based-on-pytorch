from pytorch_segnet import *
import numpy as np
import torch.utils.data as Data
import sys
import csv


device = torch.device("cuda " if torch.cuda.is_available() else "cpu")
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

        image_seg = np.zeros((1024, 2048, 3))
        image_seg = np.uint8(image_seg)
        ad_mask = np.zeros((1024, 2048, 3))
        ad_mask = np.uint8(ad_mask)
        # label = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky'
        #          ,'person','rider','car','truck','bus','train','motocycle','bicycle']

        colors = COLORS

        for c in range(CLASS_NUM):
            image_seg[:, :, 0] += np.uint8((output_np == c)) * np.uint8(colors[c][0])
            image_seg[:, :, 1] += np.uint8((output_np == c)) * np.uint8(colors[c][1])
            image_seg[:, :, 2] += np.uint8((output_np == c)) * np.uint8(colors[c][2])

        for c in range(CLASS_NUM):
            ad_mask[:, :, 0] += np.uint8((adver_output == c)) * np.uint8(colors[c][0])
            ad_mask[:, :, 1] += np.uint8((adver_output == c)) * np.uint8(colors[c][1])
            ad_mask[:, :, 2] += np.uint8((adver_output == c)) * np.uint8(colors[c][2])


        with open("./user_info.csv","w",newline='') as f:
            writer=csv.writer(f)
            for column in image_seg:
                writer.writerow(column)
        image_seg = Image.fromarray(np.uint8(image_seg))
        ad_mask = Image.fromarray(np.uint8(ad_mask))


        image_src = cv.cvtColor(image_src,cv.COLOR_BGR2RGB)
        old_image = Image.fromarray(np.uint8(image_src))

        image = Image.blend(old_image, image_seg, 0.6)
        image1 = Image.blend(old_image, ad_mask, 0.6)

        # remove the background
        # image_np = np.array(image)
        # image_np[output_np == 0] = image_src[output_np == 0]
        # image = Image.fromarray(image_np)
        image.save(OUTPUTS + path)
        image1.save(OUTPUTS1 + path)

        print(path + " is done!")


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=19, help="classes")
parser.add_argument("--weights", type=str, default="weights/SegNet_weights1625783040.5590682.pth", help="weight path")
parser.add_argument("--colors", type=int, default=[[0, 255, 0], [255, 0, 0],[0, 0, 255],[111, 74, 0],[70, 70, 70],[128, 64, 128],[0, 0, 0],[102, 102, 156],[190, 153, 153],[150, 100, 100],[107, 142, 35],
                 [152, 251, 152],[70, 130, 180],[220, 220, 0],[119, 11, 32],[215, 166, 66],[66, 88, 99],[154, 25, 244],[10, 155, 83]], help="colors")
parser.add_argument("--samples", type=str, default="samples//", help="test samples path")
parser.add_argument("--outputs", type=str, default="outputs//", help="path for save output")
parser.add_argument("--outputs1", type=str, default="outputs1//", help="path for save output")
parser.add_argument("--train_txt", type=str, default="train.txt", help="training dataset path")
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
test(SegNet)
