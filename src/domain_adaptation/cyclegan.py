import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1, 
                 ngf: int = 64, n_residual_blocks: int = 6):
        super(Generator, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, ngf, kernel_size=7, padding=3),
            nn.BatchNorm1d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.downsampling = nn.Sequential(
            nn.Conv1d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(n_residual_blocks)]
        )
        
        self.upsampling = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv1d(ngf, output_channels, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size(2)
        x = self.initial_conv(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        
        if x.size(2) != input_size:
            if x.size(2) > input_size:
                x = x[:, :, :input_size]
            else:
                pad = input_size - x.size(2)
                x = F.pad(x, [0, pad])
        
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels: int = 1, ndf: int = 64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf * 4, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    def __init__(self, input_channels: int = 1, ngf: int = 64, ndf: int = 64):
        super(CycleGAN, self).__init__()
        
        self.G_AB = Generator(input_channels, input_channels, ngf)
        self.G_BA = Generator(input_channels, input_channels, ngf)
        self.D_A = Discriminator(input_channels, ndf)
        self.D_B = Discriminator(input_channels, ndf)
        
        self.cycle_loss_weight = 10.0
        self.identity_loss_weight = 0.5

    def forward(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        
        idt_A = self.G_BA(real_A)
        idt_B = self.G_AB(real_B)
        
        return {
            'fake_B': fake_B,
            'fake_A': fake_A,
            'rec_A': rec_A,
            'rec_B': rec_B,
            'idt_A': idt_A,
            'idt_B': idt_B
        }

    def compute_loss(self, real_A: torch.Tensor, real_B: torch.Tensor, 
                     optimizer_G: torch.optim.Optimizer, 
                     optimizer_D: torch.optim.Optimizer) -> dict:
        outputs = self.forward(real_A, real_B)
        
        optimizer_G.zero_grad()
        
        identity_loss_A = F.l1_loss(outputs['idt_A'], real_A) * self.cycle_loss_weight * self.identity_loss_weight
        identity_loss_B = F.l1_loss(outputs['idt_B'], real_B) * self.cycle_loss_weight * self.identity_loss_weight
        
        cycle_loss_A = F.l1_loss(outputs['rec_A'], real_A) * self.cycle_loss_weight
        cycle_loss_B = F.l1_loss(outputs['rec_B'], real_B) * self.cycle_loss_weight
        
        fake_B_pred = self.D_B(outputs['fake_B'])
        fake_A_pred = self.D_A(outputs['fake_A'])
        
        adv_loss_G = (F.mse_loss(fake_B_pred, torch.ones_like(fake_B_pred)) + 
                      F.mse_loss(fake_A_pred, torch.ones_like(fake_A_pred))) / 2
        
        loss_G = identity_loss_A + identity_loss_B + cycle_loss_A + cycle_loss_B + adv_loss_G
        loss_G.backward(retain_graph=True)
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        
        real_A_pred = self.D_A(real_A)
        real_B_pred = self.D_B(real_B)
        
        adv_loss_D_A = (F.mse_loss(real_A_pred, torch.ones_like(real_A_pred)) + 
                        F.mse_loss(self.D_A(outputs['fake_A'].detach()), torch.zeros_like(real_A_pred))) / 2
        
        adv_loss_D_B = (F.mse_loss(real_B_pred, torch.ones_like(real_B_pred)) + 
                        F.mse_loss(self.D_B(outputs['fake_B'].detach()), torch.zeros_like(real_B_pred))) / 2
        
        loss_D = (adv_loss_D_A + adv_loss_D_B) / 2
        loss_D.backward()
        optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'adv_loss_G': adv_loss_G.item(),
            'cycle_loss': (cycle_loss_A + cycle_loss_B).item(),
            'identity_loss': (identity_loss_A + identity_loss_B).item()
        }

    def adapt_domain(self, data: torch.Tensor, direction: str = 'A2B') -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if direction == 'A2B':
                return self.G_AB(data)
            else:
                return self.G_BA(data)

    def save_models(self, path: str):
        torch.save({
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict()
        }, path)

    def load_models(self, path: str):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.G_AB.load_state_dict(checkpoint['G_AB'])
        self.G_BA.load_state_dict(checkpoint['G_BA'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])