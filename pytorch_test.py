from pytorch_segnet import *
import sys
import csv


def test(SegNet):

    SegNet.load_state_dict(torch.load(WEIGHTS,map_location=torch.device('cpu')))
    SegNet.eval()

    paths = os.listdir(SAMPLES)

    for path in paths:

        image_src = cv.imread(SAMPLES + path)
        image = cv.resize(image_src, (416, 416))

        image = image / 255.0
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = torch.unsqueeze(image, dim=0)

        output, relu_value = SegNet(image)
        output = torch.squeeze(output)
        output = output.argmax(dim=0)
        output_np = cv.resize(np.uint8(output), (2048, 1024))

        image_seg = np.zeros((1024, 2048, 3))
        image_seg = np.uint8(image_seg)
        mask = np.zeros((1024, 2048, 3))
        mask = np.uint8(mask)
        label = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky'
                 ,'person','rider','car','truck','bus','train','motocycle','bicycle']

        colors = COLORS

        for c in range(CLASS_NUM):
            image_seg[:, :, 0] += np.uint8((output_np == c)) * np.uint8(colors[c][0])
            image_seg[:, :, 1] += np.uint8((output_np == c)) * np.uint8(colors[c][1])
            image_seg[:, :, 2] += np.uint8((output_np == c)) * np.uint8(colors[c][2])


        with open("./user_info.csv","w",newline='') as f:
            writer=csv.writer(f)
            for column in image_seg:
                writer.writerow(column)
        image_seg = Image.fromarray(np.uint8(image_seg))


        image_src = cv.cvtColor(image_src,cv.COLOR_BGR2RGB)
        old_image = Image.fromarray(np.uint8(image_src))

        image = Image.blend(old_image, image_seg, 0.6)

        
        # image_np = np.array(image)
        # image_np[output_np == 0] = image_src[output_np == 0]
        # image = Image.fromarray(image_np)
        image.save(OUTPUTS + path)

        print(path + " is done!")


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=19, help="number of classes")
parser.add_argument("--weights", type=str, default="weights/SegNet_weights1625783040.5590682.pth", help="weights path")
parser.add_argument("--colors", type=int, default=[[0, 255, 0], [255, 0, 0],[0, 0, 255],[111, 74, 0],[70, 70, 70],[128, 64, 128],[0, 0, 0],[102, 102, 156],[190, 153, 153],[150, 100, 100],[107, 142, 35],
                 [152, 251, 152],[70, 130, 180],[220, 220, 0],[119, 11, 32],[215, 166, 66],[66, 88, 99],[154, 25, 244],[10, 155, 83]], help="mask")
parser.add_argument("--samples", type=str, default="samples//", help="path for sample")
parser.add_argument("--outputs", type=str, default="outputs//", help="path for output")
opt = parser.parse_args()
#print(opt)

CLASS_NUM = opt.class_num
WEIGHTS = opt.weights
COLORS = opt.colors
SAMPLES = opt.samples
OUTPUTS = opt.outputs


SegNet = SegNet(3, CLASS_NUM)
test(SegNet)
