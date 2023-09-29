from NN import *
from Format import *
from Image_process import *

layer1 = Layer(38809)
layer2 = Dense(38809,  197)
layer3 = Dense(197, 197)
layer4 = Dense(197, 2)
act1 = Activation()
act2 = Activation()
act3 = Activation()
act4 = Activation()

old = layer4.weights
for i in range(10):
    f_loss = 1.01
    f_loss_m = 100
    print(f"Epoch {i+1}")
    output1 = 0
    output2 = 0
    output3 = 0
    output4 = 0
    Dl = 0
    for j in range(100):
        img = choose(i, j)
        inputs = assign(img[0])
        outputs = img[1]
        net1 = layer1.forward(inputs)
        output1 += net1
        out1 = act1.sigmoid(net1)
        net2 = layer2.forward(out1)
        output2 += net2
        out2 = act2.ReLU(net2)
        net3 = layer3.forward(out2)
        output3 += net3
        out3 = act3.ReLU(net3)
        net4 = layer4.forward(out3)
        output4 += net4
        y_pred = act4.softmax(net4)
        print(y_pred)
        print(outputs)
        loss = Loss(y_pred, outputs)
        f_loss = loss.cross_entropy()
        print(f_loss)
        DL = loss.cross_entropy(back=1)
        print(DL)
        Dl += DL

    DA4 = act4.softmax(output4/100, back=1)
    DL4 = layer4.backward(Dl/100 * DA4)
    DA3 = act3.ReLU(output3/100, back=1)
    DL3 = layer3.backward(DL4 * DA3)
    DA2 = act2.ReLU(output2/100, back=1)
    DL2 = layer2.backward(DL3 * DA2)
    DA1 = act1.sigmoid(output1/100, back=1)

difference = layer4.weights - old
print(difference)

test_img = horz("Cat_images/cat3.jpg")
test_inputs = assign(test_img)
test1 = layer1.forward(test_inputs)
atest1 = act1.sigmoid(test1)
test2 = layer2.forward(atest1)
atest2 = act2.ReLU(test2)
test3 = layer3.forward(atest2)
atest3 = act3.ReLU(test3)
test4 = layer4.forward(atest3)
prediction = act4.softmax(test4)
print(prediction)

test_img = horz("Cat_images/cat10.jpg")
test_inputs = assign(test_img)
test1 = layer1.forward(test_inputs)
atest1 = act1.sigmoid(test1)
test2 = layer2.forward(atest1)
atest2 = act2.ReLU(test2)
test3 = layer3.forward(atest2)
atest3 = act3.ReLU(test3)
test4 = layer4.forward(atest3)
prediction = act4.softmax(test4)
print(prediction)

test_img = horz("Dog_images/dog11.jpg")
test_inputs = assign(test_img)
test1 = layer1.forward(test_inputs)
atest1 = act1.sigmoid(test1)
test2 = layer2.forward(atest1)
atest2 = act2.ReLU(test2)
test3 = layer3.forward(atest2)
atest3 = act3.ReLU(test3)
test4 = layer4.forward(atest3)
prediction = act4.softmax(test4)
print(prediction)

test_img = horz("Dog_images/dog21.jpg")
test_inputs = assign(test_img)
test1 = layer1.forward(test_inputs)
atest1 = act1.sigmoid(test1)
test2 = layer2.forward(atest1)
atest2 = act2.ReLU(test2)
test3 = layer3.forward(atest2)
atest3 = act3.ReLU(test3)
test4 = layer4.forward(atest3)
prediction = act4.softmax(test4)
print(prediction)
