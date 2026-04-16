from args import parse_train_opt

EDGE = None


def _load_edge():
    global EDGE
    if EDGE is None:
        from EDGE import EDGE as edge_cls

        EDGE = edge_cls
    return EDGE


def train(opt):
    model = _load_edge()(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        use_beats=opt.use_beats,
        beat_rep=opt.beat_rep,
        lambda_acc=opt.lambda_acc,
        lambda_beat=opt.lambda_beat,
        beat_a=opt.beat_a,
        beat_c=opt.beat_c,
        beat_estimator_ckpt=opt.beat_estimator_ckpt,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
