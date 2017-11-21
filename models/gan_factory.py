from models import gan, gan_cls, wgan_cls

class gan_factory(object):

    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'wgan':
            return wgan_cls.generator()
        elif type == 'vanilla':
            return gan.generator()


    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'wgan':
            return wgan_cls.discriminator()
        elif type == 'vanilla':
            return gan.discriminator()
