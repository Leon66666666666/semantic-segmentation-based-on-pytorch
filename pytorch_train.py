from pytorch_segnet import *


def train(SegNet):

    SegNet = SegNet.cpu()
    SegNet.load_weights(PRE_TRAINING)

    train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.SGD(SegNet.parameters(), lr=LR, momentum=MOMENTUM)

    #loss_func = nn.MSELoss
    loss_func = nn.CrossEntropyLoss(reduction='mean',ignore_index=255).cpu()

    SegNet.train()
    for epoch in range(EPOCH):
        for step, (image, label) in enumerate(train_loader):
            #print(label)
            image = image.cpu()
            label = label.cpu()
            label = label.view(BATCH_SIZE, 416, 416)

            output,relu_value = SegNet(image)



            loss = loss_func(output, torch.squeeze(label, 1).long())
            loss = loss.cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                print("Epoch:{0} || Step:{1} || Loss:{2}".format(epoch, step, format(loss, ".4f")))
            #print(relu_value)

    torch.save(SegNet.state_dict(), WEIGHTS + "SegNet_weights" + str(time.time()) + ".pth")


parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=19, help="classes")
parser.add_argument("--epoch", type=int, default=1, help="number of iteration")
parser.add_argument("--batch_size", type=int, default=1, help="batach size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--category_weight", type=float, default=[0.7502381287857225, 1.4990483912788268], help="ignore")
parser.add_argument("--train_txt", type=str, default="train.txt", help="path of dataset")
parser.add_argument("--pre_training_weight", type=str, default="vgg16_bn-6c64b313.pth", help="path for pretrain")
parser.add_argument("--weights", type=str, default="./weights/", help="path for results")
opt = parser.parse_args()
print(opt)

CLASS_NUM = opt.class_num
EPOCH = opt.epoch
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
MOMENTUM = opt.momentum
CATE_WEIGHT = opt.category_weight
TXT_PATH = opt.train_txt
PRE_TRAINING = opt.pre_training_weight
WEIGHTS = opt.weights


train_data = MyDataset(txt_path=TXT_PATH)

SegNet = SegNet(3, CLASS_NUM)
train(SegNet)
