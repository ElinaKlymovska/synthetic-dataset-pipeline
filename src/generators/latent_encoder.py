import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from typing import Optional, List
import numpy as np
from PIL import Image

class LatentEncoder:
    """
    Клас для отримання латентних векторів з фото та інтерполяції латентів для SDXL/Flux
    """
    def __init__(self, sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sdxl_model = sdxl_model
        self._load_models()

    def _load_models(self):
        # Оптимізація для CPU: використовуємо float32 замість float16
        torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
        
        # Завантажуємо тільки VAE для економії пам'яті
        self.vae = AutoencoderKL.from_pretrained(
            self.sdxl_model,
            subfolder="vae",
            torch_dtype=torch_dtype
        ).to(self.device)
        self.vae.eval()

    def image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """
        Кодує зображення у латентний простір SDXL (VAE)
        """
        # Оптимізація: зменшуємо розмір для швидшого обробки
        target_size = (512, 512)  # Зменшено з 1024x1024
        image = image.convert("RGB").resize(target_size)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[None].transpose(0, 3, 1, 2)  # BCHW
        image_tensor = torch.from_numpy(image_np).to(self.device)
        
        with torch.no_grad():
            latents = self.vae.encode(image_tensor * 2 - 1).latent_dist.sample()
            latents = latents * 0.18215  # SDXL scaling
        return latents

    def latent_to_image(self, latents: torch.Tensor) -> Image.Image:
        """
        Декодує латент у зображення через VAE
        """
        with torch.no_grad():
            latents = latents / 0.18215
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)

    def interpolate_latents(self, latents_a: torch.Tensor, latents_b: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Лінійна інтерполяція між двома латентами
        """
        return latents_a * (1 - alpha) + latents_b * alpha

    def interpolate_with_noise(self, latents: torch.Tensor, noise_level: float = 0.3) -> torch.Tensor:
        """
        Інтерполяція латенту з випадковим шумом
        """
        noise = torch.randn_like(latents)
        return self.interpolate_latents(latents, noise, noise_level)

    def batch_interpolate(self, latents: torch.Tensor, reference_latents: List[torch.Tensor], n: int = 10) -> List[torch.Tensor]:
        """
        Генерує n інтерпольованих латентів між основним латентом і списком референсів
        """
        results = []
        for ref in reference_latents:
            for alpha in np.linspace(0, 1, n):
                results.append(self.interpolate_latents(latents, ref, alpha))
        return results

    def generate_variations_fast(self, latents: torch.Tensor, num_variations: int = 8) -> List[torch.Tensor]:
        """
        Швидка генерація варіацій з різними рівнями шуму
        """
        variations = []
        noise_levels = np.linspace(0.15, 0.6, num_variations)
        
        # Генеруємо весь шум одразу для ефективності
        noise_batch = torch.randn(num_variations, *latents.shape[1:], device=self.device)
        
        for i, noise_level in enumerate(noise_levels):
            variation = self.interpolate_latents(latents, noise_batch[i], noise_level)
            variations.append(variation)
        
        return variations 