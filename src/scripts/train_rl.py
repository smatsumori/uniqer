# Logging
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)


def summarize_ep_values(
    ep_reward, ep_corr_r, ep_info_r,
    ep_prog_r, ep_opti_r, ep_turn_p, ep_disc_p, ep_desc_r,
    advantage, ep_n_step, ep_invalid_ratio, ep_related_ratio,
    ep_stop_ratio, ep_variety, ep_tk_variety, ep_n_ques, ep_valueest_loss,
    ep_policy_loss, prefix=''
):
    ep_values = {
        # rewards {CustumMetrics.label, ep_value}
        prefix+'reward': ep_reward,
        prefix+'corr_r': ep_corr_r,
        prefix+'info_r': ep_info_r,
        prefix+'prog_r': ep_prog_r,
        prefix+'opti_r': ep_opti_r,
        prefix+'turn_p': ep_turn_p,
        prefix+'disc_p': ep_disc_p,
        prefix+'desc_r': ep_desc_r,
        # advantage
        prefix+'advantage': advantage,
        # other metrics
        prefix+'n_step': ep_n_step,
        prefix+'invalid_ratio': ep_invalid_ratio,
        prefix+'related_ratio': ep_related_ratio,
        prefix+'token_variety': ep_tk_variety,
        prefix+'variety': ep_variety,
        prefix+'stop_ratio': ep_stop_ratio,

        # TODO: rename n_ques
        prefix+'n_question': ep_n_ques,
        # losses
        prefix+'valueest_loss': ep_valueest_loss,
        prefix+'policy_loss': ep_policy_loss
    }
    return ep_values


def calculate_epoch_values(eps_values, nb):
    """Calculate epoch values by averaging episode values
    by the number of epoch.

    Parameters
    ----------
    eps_values : list
        list of dict
    n_eps : int
    """
    assert len(eps_values) == nb
    epoch_values = {}
    for d in eps_values:
        for k, v in d.items():
            try:
                epoch_values[k] += v
            except KeyError:
                epoch_values[k] = v

    # average by entries
    epoch_values = {
        k: v/nb for k, v in epoch_values.items()
    }
    return epoch_values
