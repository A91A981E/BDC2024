import math


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5**epoch)}
    elif args.lradj == "type2":
        lr_adjust = {
            1: 4e-4,
            2: 2e-4,
            3: 1e-4,
            4: 1e-6,
            # 5: 1e-5,
            # 8: 1e-6,
            # 10: 5e-7,
            # 15: 1e-7,
            # 20: 5e-8,
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    elif args.lradj == "const":
        return
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))
