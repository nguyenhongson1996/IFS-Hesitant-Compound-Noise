import argparse

from src.common.consts import GlueTask, override_train_sample_ratio
from src.common.seed_utils import set_seed
from src.trainer.noisy_mtl_trainer import NoisyMTLIFSTrainer


def main():
    parser = argparse.ArgumentParser(
        description="IFS-Hesitant MTL trainer (single-phase CE-3soft)"
    )
    parser.add_argument("--model", default="bert-base-uncased",
                        help="Pretrained encoder.")
    parser.add_argument("--model_name", default="noisy_ifs_mrpc_rte_cola",
                        help="Name used when saving model + logs.")
    parser.add_argument("--tasks", nargs="+", default=["MRPC", "RTE", "COLA"],
                        help="GLUE tasks to train on (3-letter task names).")
     # Compound task-class noise model (paper Sec. 2)
    parser.add_argument("--epsilon_t", type=float, default=0.2,
                        help="Task-id noise rate (paper eps_t).")
    parser.add_argument("--class_noise_rho", type=float, default=0.0,
                        help="Class-flip prob conditional on task-noisy (paper rho).")
    parser.add_argument("--class_noise_rho_indep", type=float, default=0.0,
                        help="Independent class-flip prob (paper rho_indep).")
    parser.add_argument("--noise_seed", type=int, default=42)
     # IFS-Hesitant hyperparameters (paper Sec. 4.2, defaults match the paper)
    parser.add_argument("--w_min", type=float, default=0.1,
                        help="Trust-weight floor after batch-mean normalisation "
                             "(paper Eq. wmin, default 0.1).")
    parser.add_argument("--weight_norm", choices=["mean", "none"], default="mean",
                        help="Batch-mean trust-weight normalisation mode. "
                             "'mean' (default) rescales w to weight_norm_target. "
                             "'none' disables batch normalisation entirely "
                             "(matches small-arch --weight_norm none, used in "
                             "the 2026-05-18 stabilizer ablation).")
    parser.add_argument("--weight_norm_target", type=float, default=0.5,
                        help="Target batch-mean of w before clamp "
                             "(paper Eq. weight_norm, default 0.5).")
    parser.add_argument("--lambda_other", type=float, default=1.0,
                        help="Barrier weight on non-observed tasks (paper Sec. 4.2, "
                             "default 1.0).")
    parser.add_argument("--lambda_pi", type=float, default=1.0,
                        help="Weight on the pi-loss combined with class CE "
                             "(effective lambda in Eq. warmup is lambda_pi * lambda_other; "
                             "both default to 1.0).")
    parser.add_argument("--trust_signal",
                        choices=["ifs", "entropy", "margin", "loss"],
                        default="ifs",
                        help="Trust-signal choice (paper Eq. trust_weight). "
                             "'ifs' reproduces the paper. Others are for the "
                             "trust-signal ablation only.")
    parser.add_argument("--class_loss",
                        choices=["kl", "hamming"],
                        default="kl",
                        help="IFS-divergence for observed-head class evidence. "
                             "'kl' (default) reproduces paper -log sat. "
                             "'hamming' uses 1 - sat (bounded, class-noise robust).")
    parser.add_argument("--l_pred_alpha", type=float, default=0.0,
                        help="L_pred branch weight (0=disabled, reproduces paper; "
                             "1=full self-supervised low-trust branch).")
    parser.add_argument("--l_pred_barrier", action="store_true",
                        help="Include the barrier on heads != t_pred in L_pred "
                             "(literal old two-phase form). Default False = "
                             "class-only L_pred.")
    parser.add_argument("--head_type", choices=["ifs", "softmax3"],
                        default="ifs",
                        help="'softmax3' (paper headline): "
                             "(mu,pi,nu)=softmax(m,n,p) over the three "
                             "logits per task. 'ifs' (default for "
                             "back-compat; parameterisation ablation): "
                             "factored head with (mu_tilde,nu_tilde)=softmax(m,n), "
                             "pi=sigma(p), mu=(1-pi)*mu_tilde, nu=(1-pi)*nu_tilde.")
    parser.add_argument("--tnorm",
                        choices=["product", "min", "lukasiewicz", "hamacher"],
                        default="product",
                        help="T-norm for the trust gate w=T(1-pi_t_tilde, sat_t_tilde(y_tilde)). "
                             "'product' (default, paper headline). 'min' "
                             "(Godel), 'lukasiewicz' (nilpotent, has zero "
                             "divisors), 'hamacher' (gamma=2 Hamacher product) "
                             "are for the P1 T-norm ablation.")
    parser.add_argument("--bayes_trust", action="store_true",
                        help="Use the Sec. 4.2-consistent trust weight "
                             "w = T(1-pi_obs, sat_tilde_obs) where sat_tilde is the "
                             "conditional class probability from "
                             "softmax(m,n) directly (factored head only). "
                             "Default formula w=T(1-pi, sat) gives "
                             "(1-pi)^2*sat_tilde in factored mode, which doesn't "
                             "match the prose's 'product of two posteriors' "
                             "interpretation.")
    parser.add_argument("--trust_no_gate", action="store_true",
                        help="Ablation: replace trust weight w with constant 1.0 "
                             "(reproduces L_sup-only branch of "
                             "tab:loss_ablation). Bypasses weight_norm + w_min.")
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_gradient_clip", type=float, default=1.0)
    # Misc
    parser.add_argument("--sst2_ratio", type=float, default=None,
                        help="Override SST-2 train subsampling ratio.")
    parser.add_argument("--out_dir", default="noisy_ifs_mrpc_rte_cola",
                        help="Where to save model checkpoints + eval logs.")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches per epoch (smoke testing).")
    args = parser.parse_args()

    if args.sst2_ratio is not None:
        override_train_sample_ratio(GlueTask.SST2, args.sst2_ratio)
    set_seed(args.noise_seed)

    trainer = NoisyMTLIFSTrainer(
        model_name_or_path=args.model,
        task_name=args.model_name,
        tasks=[GlueTask[t] for t in args.tasks],
        epsilon_t=args.epsilon_t,
        noise_seed=args.noise_seed,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
        do_blindly_decode=True,
        w_min=args.w_min,
        weight_norm=args.weight_norm,
        weight_norm_target=args.weight_norm_target,
        lambda_other=args.lambda_other,
        lambda_pi=args.lambda_pi,
        trust_signal=args.trust_signal,
        class_loss=args.class_loss,
        l_pred_alpha=args.l_pred_alpha,
        l_pred_barrier=args.l_pred_barrier,
        head_type=args.head_type,
        tnorm=args.tnorm,
        bayes_trust=args.bayes_trust,
        trust_no_gate=args.trust_no_gate,
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
