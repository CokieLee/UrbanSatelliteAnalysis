import torch


def get_mean_stdev(loader):
    # Compute mean and stdev of all pixels in dataset
    num_images = 0
    mean = torch.zeros(3)
    var = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.sum(dim=[0,2,3])

    mean = mean/num_pixels

    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        var += ((images - mean[None, :, None, None])**2).sum(dim=[0,2,3])

    var = var/num_pixels
    stdev = torch.sqrt(var)
    return mean, stdev

mean, stdev = get_mean_stdev(total_loader)
print(f'mean: {mean}, stdev: {stdev}')





