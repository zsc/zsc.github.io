import unittest
import torch
from model_unet import UNet, DownBlock, UpBlock, TimeEmbedding, ResidualBlock

class TestUNetSkipConnections(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_emb_dim = 64
        self.time_embedding = TimeEmbedding(self.time_emb_dim).to(self.device)

    def test_down_up_block_concat(self):
        print("\nTesting DownBlock and UpBlock skip connection concatenation...")
        batch_size = 2
        image_size = 16 # Use power of 2 for easier tracking
        
        down_in_ch = 16
        down_out_ch = 32 # This will be the skip connection channels and DownBlock output channels
        
        down_block = DownBlock(in_channels=down_in_ch, out_channels=down_out_ch, time_channels=self.time_emb_dim).to(self.device)
        
        dummy_x_down = torch.randn(batch_size, down_in_ch, image_size, image_size, device=self.device)
        dummy_t_val = torch.randint(0, 100, (batch_size,), device=self.device).float()
        dummy_t_emb = self.time_embedding(dummy_t_val)

        # Pass through DownBlock
        # down_output is the spatially reduced feature map, skip_conn_tensor is from before spatial reduction
        down_output, skip_conn_tensor = down_block(dummy_x_down, dummy_t_emb)
        
        self.assertEqual(skip_conn_tensor.shape[1], down_out_ch, "Skip connection channel count mismatch.")
        self.assertEqual(down_output.shape[1], down_out_ch, "DownBlock output (to be upsampled) channel count mismatch.")
        self.assertEqual(down_output.shape[2], image_size // 2, "DownBlock output spatial dim mismatch.")

        # UpBlock setup:
        # x_channels: channels of the tensor to be upsampled (i.e., down_output channels)
        upblock_x_channels = down_output.shape[1] 
        # skip_channels: channels of the skip connection (i.e., skip_conn_tensor channels)
        upblock_skip_channels = skip_conn_tensor.shape[1]
        # out_channels: desired output channels of the UpBlock's ResidualBlock
        upblock_out_channels = down_out_ch # Example: make it same as skip/down_out_ch

        up_block = UpBlock(x_channels=upblock_x_channels, 
                           skip_channels=upblock_skip_channels, 
                           out_channels=upblock_out_channels, 
                           time_channels=self.time_emb_dim).to(self.device)
        
        # Check if UpBlock's upsampler is configured to accept down_output's channels
        self.assertEqual(down_output.shape[1], up_block.upsample.in_channels,
                         f"Channel mismatch: DownBlock output ({down_output.shape[1]}) "
                         f"vs. UpBlock upsampler configured input channels ({up_block.upsample.in_channels}).")
        
        # Pass through UpBlock
        # `down_output` is the tensor that would be fed into `up_block.upsample`.
        up_output = up_block(down_output, skip_conn_tensor, dummy_t_emb)

        self.assertEqual(up_output.shape[1], upblock_out_channels, "UpBlock output channel count mismatch.")
        self.assertEqual(up_output.shape[2], image_size, "UpBlock output spatial dim mismatch.") # Assumes upsample doubles size
        print("DownBlock and UpBlock skip connection test passed.")

    def test_unet_full_pass_channel_config(self):
        print("\nTesting U-Net full pass with specific channel configuration...")
        batch_size = 2
        image_size = 14 # MNIST size
        image_channels = 1
        n_channels = 16 # Base channels
        ch_mults = (1, 2) # e.g., 16 -> 32
        time_emb_dim = self.time_emb_dim
        
        # With ch_mults=(1,2) and n_channels=16:
        # After UpBlocks, current_channels will be n_channels * ch_mults[0] = 16 * 1 = 16.
        # final_norm is nn.GroupNorm(8, 16) - This matches.
        # final_conv is Conv2d(16, image_channels, ...) - This matches.

        model = UNet(image_channels=image_channels, n_channels=n_channels, 
                     ch_mults=ch_mults, time_emb_dim=time_emb_dim).to(self.device)
        
        dummy_x = torch.randn(batch_size, image_channels, image_size, image_size, device=self.device)
        dummy_t_val = torch.randint(0, 100, (batch_size,), device=self.device).float()
        
        try:
            output = model(dummy_x, dummy_t_val)
            self.assertEqual(output.shape, dummy_x.shape)
            print("U-Net full pass test successful with configured channels.")
        except RuntimeError as e:
            self.fail(f"U-Net forward pass failed with channel configuration. Error: {e}")

    def test_unet_final_layer_channels(self):
        print("\nTesting U-Net final layer channel consistency with ch_mults[0] != 1...")
        batch_size = 1
        image_size = 16
        image_channels = 1
        n_channels = 8
        ch_mults = (2, 4) # Base channels for final norm/conv should be n_channels * ch_mults[0] = 8 * 2 = 16
        time_emb_dim = self.time_emb_dim

        model = UNet(image_channels=image_channels, n_channels=n_channels,
                     ch_mults=ch_mults, time_emb_dim=time_emb_dim).to(self.device)

        # Check final_norm and final_conv input channels
        expected_final_block_in_channels = n_channels * ch_mults[0]
        self.assertEqual(model.final_norm.num_channels, expected_final_block_in_channels,
                         "final_norm input channels mismatch")
        self.assertEqual(model.final_conv.in_channels, expected_final_block_in_channels,
                         "final_conv input channels mismatch")
        
        dummy_x = torch.randn(batch_size, image_channels, image_size, image_size, device=self.device)
        dummy_t_val = torch.randint(0, 100, (batch_size,), device=self.device).float()
        try:
            output = model(dummy_x, dummy_t_val)
            self.assertEqual(output.shape, dummy_x.shape)
            print("U-Net final layer channel test successful.")
        except RuntimeError as e:
            self.fail(f"U-Net forward pass failed in final layer channel test. Error: {e}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
