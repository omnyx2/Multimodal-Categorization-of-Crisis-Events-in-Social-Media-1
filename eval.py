def eval():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    classnum = 9
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)
        
    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)
#    
    recall = (recall.numpy()[0]*100).round(3)
    precision = (precision.numpy()[0]*100).round(3)
    F1 = (F1.numpy()[0]*100).round(3)
    accuracy = (accuracy.numpy()[0]*100).round(3)