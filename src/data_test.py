
#    f = open('data/train_test/train_x_real_filter.txt', 'r')
#    f1 = open('data/train_test/train_y_real_filter.txt', 'r')
#    X_test_data = f.read()
#    Y_test_data = f1.read()
#    f.close()
#    f1.close()
#    test_x = [text_to_word_sequence(x_)[::-1] for x_ in X_test_data.split('\n') if
#             len(x_.split(' ')) > 0 and len(x_.split(' ')) <= MAX_LEN]
#    test_y = [text_to_word_sequence(y_)[::-1] for y_ in Y_test_data.split('\n') if
#             len(y_.split(' ')) > 0 and len(y_.split(' ')) <= MAX_LEN]
#
#    X_max_test = max(map(len, test_x))
#    for index in range(len(test_x)):
#        round = X_max_test - len(test_x[index])
#        while round:
#            test_x[index].append('.')
#            test_y[index].append('O')
#            round -= 1
#
#    for i, sentence in enumerate(test_x):
#        for j, word in enumerate(sentence):
#            if word in X_word_to_ix:
#                test_x[i][j] = X_word_to_ix[word]
#            else:
#                test_x[i][j] = X_word_to_ix['UNK']
#
#    for i, sentence in enumerate(test_y):
#        for j, word in enumerate(sentence):
#            if word in y_word_to_ix:
#                test_y[i][j] = y_word_to_ix[word]
#            else:
#                test_y[i][j] = y_word_to_ix['UNK']


# p1 = list(model.parameters())[0].clone()
# optimizer.step()
# p2 = list(model.parameters())[0].clone()
# print(torch.equal(p1,p2))
#            if count % 100 == 0:
#                # torch.save(model, '/Users/test/Desktop/RE/model')
#                print("{0} epoch , current training loss {1} : ".format(epoch, loss.data))
#                log.write(str(epoch) + "epoch" + "current trainning loss : " + str(loss.data))
#                test_scores, test_targets = predict(
#                    test_x[0:BATCH_SIZE],
#                    test_y[0:BATCH_SIZE],
#                    model,
#                )
#                loss_test = loss_function(test_scores, test_targets)
#                print(".............current test loss............ {} : ".format(loss_test/BATCH_SIZE))
#                log.write("current test loss : " + str(loss_test/BATCH_SIZE))
#            count += 1
