from utils.rl.rl_utils import CustomMetrics as CM


def init_metrics(args):
    """ Register custom metrics
    """
    rr = args.running_reward
    rd = args.running_discount

    # TODO: refactoring gross.
    # --- Training Metrics  (Episode Metrics) ---
    # rewards
    met_reward = CM('reward', 'Rewards', rr)
    met_corr_r = CM('corr_r', 'Correct Rewards', rr)
    met_info_r = CM('info_r', 'Informativeness Rewards', rr, print_log=False)
    met_prog_r = CM('prog_r', 'Progressive Rewards', rr, print_log=False)
    met_opti_r = CM(
        'opti_r', 'Optimize Number of Question Reward', rr, print_log=False)
    met_turn_p = CM('turn_p', 'Turn Taking Penalty', rr, print_log=False)
    met_disc_p = CM(
        'disc_p', 'Turn Taking Penalty (Discounted)', rr, print_log=False
    )
    met_desc_r = CM('desc_r', 'Descriptive Rewards', rr, print_log=False)

    # training metrics
    met_advantage = CM('advantage', 'Advantage Value', rd, print_log=False)
    met_variety = CM('variety', 'Variety of questions', rd, print_log=False)
    met_n_step = CM('n_step', 'Number of steps', rd, print_log=False)

    met_valueest_loss = CM(
        'valueest_loss', 'Baseline Function Loss', rd, print_log=False
    )
    met_policy_loss = CM(
        'policy_loss', 'Policy Function Loss', rd, print_log=False
    )

    # Question quality metrics
    met_invalid_ratio = CM(
        'invalid_ratio', 'Invalid question ratio', rd, print_log=False
    )
    met_related_ratio = CM(
        'related_ratio', 'Related question ratio', rd, print_log=False
    )
    met_n_question = CM(
        'n_question', 'Number of question', rd, print_log=False
    )
    met_stop_ratio = CM(
        'stop_ratio', 'Successfully submitted ratio', rd, print_log=False
    )
    met_token_variety = CM(
        'token_variety', 'The variety of tokens for the questions', rd,
        print_log=False
    )

    # --- Validation Metrics (Epoch Metrics) ---
    val_met_reward = CM(
        'val_reward', 'Rewards', rr, met_epoch=True, split='valid'
    )
    val_met_corr_r = CM(
        'val_corr_r', 'Correct Rewards', rr, met_epoch=True, split='valid'
    )
    val_met_info_r = CM(
        'val_info_r', 'Informativeness Rewards', rr,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_prog_r = CM(
        'val_prog_r', 'Progressive Rewards', rr,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_opti_r = CM(
        'val_opti_r', 'Optimize Number of Question Reward', rr,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_turn_p = CM(
        'val_turn_p', 'Turn Taking Penalty', rr,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_disc_p = CM(
        'val_disc_p', 'Turn Taking Penalty (Discounted)', rr,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_desc_r = CM(
        'val_desc_r', 'Descriptive Rewards', rr,
        print_log=False, met_epoch=True, split='valid'
    )

    val_met_variety = CM(
        'val_variety', 'Variety of questions', rd,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_n_step = CM(
        'val_n_step', 'Number of steps', rd,
        print_log=False, met_epoch=True, split='valid'
    )

    val_met_policy_loss = CM(
        'val_policy_loss', 'Policy Function Loss', rd,
        print_log=False, met_epoch=True, split='valid'
    )

    # Question quality metrics
    val_met_invalid_ratio = CM(
        'val_invalid_ratio', 'Invalid question ratio', rd,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_related_ratio = CM(
        'val_related_ratio', 'Related question ratio', rd,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_n_question = CM(
        'val_n_question', 'Number of questions', rd,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_stop_ratio = CM(
        'val_stop_ratio', 'Successfully submitted ratio', rd,
        print_log=False, met_epoch=True, split='valid'
    )
    val_met_token_variety = CM(
        'val_token_variety', 'The variety of tokens for the questions',
        rd, print_log=False, met_epoch=True, split='valid'
    )

    # --- Test Metrics (Epoch Metrics, object mode) ---
    test_o_met_reward = CM(
        'test_o_reward', 'Rewards', rr, met_epoch=True,
        split='test_o'
    )
    test_o_met_corr_r = CM(
        'test_o_corr_r', 'Correct Rewards', rr, met_epoch=True,
        split='test_o', track_met='val_corr_r'
    )
    test_o_met_info_r = CM(
        'test_o_info_r', 'Informativeness Rewards', rr,
        print_log=False, met_epoch=True, split='test_o'
    )
    test_o_met_prog_r = CM(
        'test_o_prog_r', 'Progressive Rewards', rr,
        print_log=False, met_epoch=True, split='test_o'
    )
    test_o_met_opti_r = CM(
        'test_o_opti_r', 'Optimize Number of Question Reward', rr,
        print_log=False, met_epoch=True, split='test_o'
    )
    test_o_met_turn_p = CM(
        'test_o_turn_p', 'Turn Taking Penalty', rr,
        print_log=False, met_epoch=True, split='test_o'
    )
    test_o_met_disc_p = CM(
        'test_o_disc_p', 'Turn Taking Penalty (Discounted)', rr,
        print_log=False, met_epoch=True, split='test_o'
    )
    test_o_met_desc_r = CM(
        'test_o_desc_r', 'Descriptive Rewards', rr,
        print_log=False, met_epoch=True, split='test_o'
    )

    test_o_met_variety = CM(
        'test_o_variety', 'Variety of questions', rd,
        print_log=False, met_epoch=True, split='test_o'
    )
    test_o_met_n_step = CM(
        'test_o_n_step', 'Number of steps', rd,
        print_log=False, met_epoch=True, split='test_o'
    )

    test_o_met_policy_loss = CM(
        'test_o_policy_loss', 'Policy Function Loss', rd,
        print_log=False, met_epoch=True, split='test_o'
    )

    # Question quality metrics
    test_o_met_invalid_ratio = CM(
        'test_o_invalid_ratio', 'Invalid question ratio', rd,
        print_log=False, met_epoch=True, split='test_o',
        track_met='val_corr_r'
    )
    test_o_met_related_ratio = CM(
        'test_o_related_ratio', 'Related question ratio', rd,
        print_log=False, met_epoch=True, split='test_o',
        track_met='val_corr_r'
    )
    test_o_met_n_question = CM(
        'test_o_n_question', 'Number of questions', rd,
        print_log=False, met_epoch=True, split='test_o',
        track_met='val_corr_r'
    )
    test_o_met_stop_ratio = CM(
        'test_o_stop_ratio', 'Successfully submitted ratio', rd,
        print_log=False, met_epoch=True, split='test_o',
        track_met='val_corr_r'
    )
    test_o_met_token_variety = CM(
        'test_o_token_variety', 'The variety of tokens for the questions',
        rd, print_log=False, met_epoch=True, split='test_o',
        track_met='val_corr_r'
    )

    # --- Test Metrics (Epoch Metrics, object mode) ---
    test_i_met_reward = CM(
        'test_i_reward', 'Rewards', rr, met_epoch=True,
        split='test_i'
    )
    test_i_met_corr_r = CM(
        'test_i_corr_r', 'Correct Rewards', rr, met_epoch=True,
        split='test_i', track_met='val_corr_r'
    )
    test_i_met_info_r = CM(
        'test_i_info_r', 'Informativeness Rewards', rr,
        print_log=False, met_epoch=True, split='test_i'
    )
    test_i_met_prog_r = CM(
        'test_i_prog_r', 'Progressive Rewards', rr,
        print_log=False, met_epoch=True, split='test_i'
    )
    test_i_met_opti_r = CM(
        'test_i_opti_r', 'Optimize Number of Question Reward', rr,
        print_log=False, met_epoch=True, split='test_i'
    )
    test_i_met_turn_p = CM(
        'test_i_turn_p', 'Turn Taking Penalty', rr,
        print_log=False, met_epoch=True, split='test_i'
    )
    test_i_met_disc_p = CM(
        'test_i_disc_p', 'Turn Taking Penalty (Discounted)', rr,
        print_log=False, met_epoch=True, split='test_i'
    )
    test_i_met_desc_r = CM(
        'test_i_desc_r', 'Descriptive Rewards', rr,
        print_log=False, met_epoch=True, split='test_i'
    )

    test_i_met_variety = CM(
        'test_i_variety', 'Variety of questions', rd,
        print_log=False, met_epoch=True, split='test_i'
    )
    test_i_met_n_step = CM(
        'test_i_n_step', 'Number of steps', rd,
        print_log=False, met_epoch=True, split='test_i'
    )

    test_i_met_policy_loss = CM(
        'test_i_policy_loss', 'Policy Function Loss', rd,
        print_log=False, met_epoch=True, split='test_i'
    )

    # Question quality metrics
    test_i_met_invalid_ratio = CM(
        'test_i_invalid_ratio', 'Invalid question ratio', rd,
        print_log=False, met_epoch=True, split='test_i',
        track_met='val_corr_r'
    )
    test_i_met_related_ratio = CM(
        'test_i_related_ratio', 'Related question ratio', rd,
        print_log=False, met_epoch=True, split='test_i',
        track_met='val_corr_r'
    )
    test_i_met_n_question = CM(
        'test_i_n_question', 'Number of questions', rd,
        print_log=False, met_epoch=True, split='test_i',
        track_met='val_corr_r'
    )
    test_i_met_stop_ratio = CM(
        'test_i_stop_ratio', 'Successfully submitted ratio', rd,
        print_log=False, met_epoch=True, split='test_i',
        track_met='val_corr_r'
    )
    test_i_met_token_variety = CM(
        'test_i_token_variety', 'The variety of tokens for the questions',
        rd, print_log=False, met_epoch=True, split='test_i',
        track_met='val_corr_r'
    )

    mets = [
        # Train metrics
        met_reward, met_corr_r, met_info_r, met_prog_r,
        met_opti_r, met_turn_p, met_disc_p, met_desc_r,
        met_advantage, met_variety, met_token_variety, met_n_step,
        met_valueest_loss, met_policy_loss,
        met_invalid_ratio, met_related_ratio,
        met_n_question, met_stop_ratio,

        # Validation metrics
        val_met_reward, val_met_corr_r, val_met_info_r, val_met_prog_r,
        val_met_opti_r, val_met_turn_p, val_met_disc_p, val_met_desc_r,
        val_met_variety, val_met_token_variety, val_met_n_step,
        val_met_policy_loss,
        val_met_invalid_ratio, val_met_related_ratio,
        val_met_n_question, val_met_stop_ratio,

        # Test metrics (new object)
        test_o_met_reward, test_o_met_corr_r, test_o_met_info_r,
        test_o_met_prog_r,
        test_o_met_opti_r, test_o_met_turn_p, test_o_met_disc_p,
        test_o_met_desc_r,
        test_o_met_variety, test_o_met_token_variety, test_o_met_n_step,
        test_o_met_policy_loss,
        test_o_met_invalid_ratio, test_o_met_related_ratio,
        test_o_met_n_question, test_o_met_stop_ratio,

        # Test metrics (new image)
        test_i_met_reward, test_i_met_corr_r, test_i_met_info_r,
        test_i_met_prog_r,
        test_i_met_opti_r, test_i_met_turn_p, test_i_met_disc_p,
        test_i_met_desc_r,
        test_i_met_variety, test_i_met_token_variety, test_i_met_n_step,
        test_i_met_policy_loss,
        test_i_met_invalid_ratio, test_i_met_related_ratio,
        test_i_met_n_question, test_i_met_stop_ratio,
    ]
    return mets
