import argparse

from src.common.consts import GlueTask, override_train_sample_ratio
from src.common.seed_utils import set_seed
from src.trainer.fqbert_trainer import FQBertNoisyTrainer


def main():
    parser = argparse.ArgumentParser(
        description="FQ-IFS: FQ-BERT on compound task-class noise.")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--model_name", default="baseline_fqbert")
    parser.add_argument("--tasks", nargs="+", default=["MRPC", "RTE", "COLA"])
    parser.add_argument("--epsilon_t", type=float, default=0.3,
                        help="Task-id noise rate (paper eps_t).")
    parser.add_argument("--class_noise_rho", type=float, default=1.0,
                        help="Class-flip prob conditional on task-noisy (paper rho).")
    parser.add_argument("--class_noise_rho_indep", type=float, default=0.0,
                        help="Independent class-flip prob (paper rho_indep).")
    parser.add_argument("--sst2_ratio", type=float, default=None)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_gradient_clip", type=float, default=1.0)
    parser.add_argument("--lambda_hes", type=float, default=1.0,
                        help="Weight on the hesitation loss (FQ-BERT lambda, Sec. 4.3).")
    parser.add_argument("--out_dir", default="results/m1/baseline_fqbert")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches per epoch (smoke test).")
    args = parser.parse_args()

    if args.sst2_ratio is not None:
        override_train_sample_ratio(GlueTask.SST2, args.sst2_ratio)
    set_seed(args.noise_seed)

    trainer = FQBertNoisyTrainer(
        model_name_or_path=args.model,
        task_name=args.model_name,
        tasks=[GlueTask[t] for t in args.tasks],
        epsilon_t=args.epsilon_t,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
        warmup_epochs=args.num_epochs,
        noise_seed=args.noise_seed,
        do_blindly_decode=True,
        lambda_hes=args.lambda_hes,
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
