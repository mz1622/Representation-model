"""Run the prepared no-optimization fit diagnostic on the audited recovered model."""
import diagnose_foodnutrigpt_r9_capacity256_fit as fit

fit.RUN = fit.ROOT/'output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1'
if __name__ == '__main__':
    fit.main()
