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
        beat_loss_cap_mode=opt.beat_loss_cap_mode,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        resume_training_state=bool(opt.checkpoint) and not opt.finetune_from_checkpoint,
        motion_format=opt.motion_format,
        lambda_g1_fk=getattr(opt, "lambda_g1_fk", 0.0),
        lambda_g1_fk_vel=getattr(opt, "lambda_g1_fk_vel", 0.0),
        lambda_g1_fk_acc=getattr(opt, "lambda_g1_fk_acc", 0.0),
        lambda_g1_foot=getattr(opt, "lambda_g1_foot", 0.0),
        lambda_g1_kin=getattr(opt, "lambda_g1_kin", 1.0),
        g1_kin_loss_warmup_epochs=getattr(opt, "g1_kin_loss_warmup_epochs", 0),
        g1_kin_loss_max_fraction=getattr(opt, "g1_kin_loss_max_fraction", 0.0),
        g1_fk_model_path=getattr(
            opt,
            "g1_fk_model_path",
            "third_party/unitree_g1_description/g1_29dof_rev_1_0.xml",
        ),
        g1_root_quat_order=getattr(opt, "g1_root_quat_order", "xyzw"),
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
