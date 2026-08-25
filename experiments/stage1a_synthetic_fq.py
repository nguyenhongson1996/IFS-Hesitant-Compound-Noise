import argparse

from src.common.seed_utils import set_seed
from src.data_processors.synthetic_dataset import tasks_for, input_dim_for
from src.trainer.synthetic_fq_trainer import SyntheticFQTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon_t", type=float, default=0.3)
    parser.add_argument("--class_noise_rho", type=float, default=0.0)
    parser.add_argument("--class_noise_rho_indep", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--encoder_output_dim", type=int, default=32)
    parser.add_argument("--lambda_hes", type=float, default=1.0,
                        help="FQ hesitation barrier weight (lambda in Sec. 4.3).")
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/stage1a/synthetic_fq")
    parser.add_argument("--model_name", default=None,
                        help="Run identifier (unused; queue compatibility).")
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.noise_seed)

    trainer = SyntheticFQTrainer(
        tasks=tasks_for(args.num_tasks),
        epsilon_t=args.epsilon_t,
        noise_seed=args.noise_seed,
        input_dim=input_dim_for(args.num_tasks),
        hidden_dims=args.hidden_dims,
        encoder_output_dim=args.encoder_output_dim,
        lambda_hes=args.lambda_hes,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
    )
    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        n_per_class=args.n_per_class,
        sigma=args.sigma,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
