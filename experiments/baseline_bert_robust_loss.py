import argparse

from src.common.consts import GlueTask, override_train_sample_ratio
from src.common.seed_utils import set_seed
from src.trainer.bert_robust_loss_trainer import BertRobustLossTrainer


def main():
    parser = argparse.ArgumentParser(
        description="BERT-scale robust-loss baselines (GCE / Bootstrapping)")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--model_name", default="baseline_bert_robust_loss")
    parser.add_argument("--tasks", nargs="+", default=["MRPC", "RTE", "COLA"])
    parser.add_argument("--loss_type", choices=["ce", "gce", "bootstrapping"],
                        default="ce")
    parser.add_argument("--gce_q", type=float, default=0.7)
    parser.add_argument("--bootstrap_beta", type=float, default=0.95)
    parser.add_argument("--epsilon_t", type=float, default=0.3)
    parser.add_argument("--class_noise_rho", type=float, default=0.0)
    parser.add_argument("--class_noise_rho_indep", type=float, default=0.0)
    parser.add_argument("--sst2_ratio", type=float, default=None)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_gradient_clip", type=float, default=1.0)
    parser.add_argument("--lambda_other", type=float, default=1.0,
                        help="Unused (CLI compat).")
    parser.add_argument("--out_dir", default="results/bert_robust_loss")
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    if args.sst2_ratio is not None:
        override_train_sample_ratio(GlueTask.SST2, args.sst2_ratio)
    set_seed(args.noise_seed)

    trainer = BertRobustLossTrainer(
        model_name_or_path=args.model,
        task_name=args.model_name,
        tasks=[GlueTask[t] for t in args.tasks],
        epsilon_t=args.epsilon_t,
        warmup_epochs=args.num_epochs,
        noise_seed=args.noise_seed,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
        do_blindly_decode=True,
        lambda_other=args.lambda_other,
        loss_type=args.loss_type,
        gce_q=args.gce_q,
        bootstrap_beta=args.bootstrap_beta,
    )
    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_gradient_clip=args.max_gradient_clip,
        out_dir=args.out_dir,
        lr=1e-4,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
