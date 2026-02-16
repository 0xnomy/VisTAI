import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, student_dim, teacher_dim, hidden_dim=1024):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, teacher_dim)
        )
    
    def forward(self, x):
        return self.projection(x)
