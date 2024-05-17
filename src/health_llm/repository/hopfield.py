import torch.nn as nn
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
import torch.nn.functional as F
import torch


class HopfieldRetrievalModel(nn.Module):
    def __init__(self, beta=0.125, update_steps_max=3):
        # def __init__(self, beta=0.125):
        super(HopfieldRetrievalModel, self).__init__()
        self.hopfield = Hopfield(
            scaling=beta,
            update_steps_max=update_steps_max,
            update_steps_eps=1e-5,
            # do not project layer input
            state_pattern_as_static=True,
            stored_pattern_as_static=True,
            pattern_projection_as_static=True,
            # do not pre-process layer input
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            # do not post-process layer output
            disable_out_projection=True,
        )

    def forward(self, memory, trg):
        memory = torch.unsqueeze(memory, 0)
        trg = torch.unsqueeze(trg, 0)
        output = self.hopfield((memory, trg, memory))
        output = output.squeeze(0)
        memories = memory.squeeze(0)
        # temp = torch.bmm(F.softmax(attn_output_weights_init, dim=-1), memory).squeeze(0)
        pair_list = F.normalize(output) @ F.normalize(memories).t()  # step1
        return pair_list


# if __name__ == "__main__":
#     reports = read_reports(
#         "/Users/chongzhang/PycharmProjects/ai_for_health_final/dataset_folder/health_report_{243}"
#     )  # 13452
#     know = retrieval_info(
#         reports, "/Users/chongzhang/PycharmProjects/ai_for_health_final/", 3
#     )
#     for i in know:
#         print(i)
