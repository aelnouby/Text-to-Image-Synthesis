from models import gan, gan_cls, wgan_cls, wgan

class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'wgan':
            return wgan_cls.generator()
        elif type == 'vanilla_gan':
            return gan.generator()
        elif type == 'vanilla_wgan':
            return wgan.generator()

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'wgan':
            return wgan_cls.discriminator()
        elif type == 'vanilla_gan':
            return gan.discriminator()
        elif type == 'vanilla_wgan':
            return wgan.discriminator()
