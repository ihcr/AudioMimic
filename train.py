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
        learning_rate_was_explicit=getattr(opt, "learning_rate_was_explicit", False),
        weight_decay=opt.weight_decay,
        use_beats=opt.use_beats,
        beat_rep=opt.beat_rep,
        lambda_acc=opt.lambda_acc,
        lambda_beat=opt.lambda_beat,
        beat_a=opt.beat_a,
        beat_c=opt.beat_c,
        beat_estimator_ckpt=opt.beat_estimator_ckpt,
        beat_estimator_max_val_loss=opt.beat_estimator_max_val_loss,
        beat_loss_start_epoch=opt.beat_loss_start_epoch,
        beat_loss_warmup_epochs=opt.beat_loss_warmup_epochs,
        beat_loss_max_fraction=opt.beat_loss_max_fraction,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        resume_training_state=bool(opt.checkpoint) and not opt.finetune_from_checkpoint,
        motion_format=opt.motion_format,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
