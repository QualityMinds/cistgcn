from .. import models as net


def choose_net(architecture, opt):
    if architecture == "CISTGCN_eval":
        model = net.CISTGCN_eval(opt.architecture_config, opt.learning_config).cuda()
    elif architecture == "CISTGCN_0":
        model = net.CISTGCN_0(opt.architecture_config, opt.learning_config).cuda()
    else:
        raise ValueError("Network Architecture you are trying to call does not exist in our Repository ;)")
    return model
