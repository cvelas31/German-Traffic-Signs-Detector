import click


@click.group()
def main():
    pass

@main.command("download")
def download():
    print("here I should be downloading the data...")
    
@main.command("train")
@click.option('-m', '--model')
@click.option('-d', '--directory')
def train(model, directory):
    msg = "I should be training model {} with data from directory {}"
    print(msg.format(model, directory))

@main.command("test")
@click.option('-m', '--model')
@click.option('-d', '--directory')
def test(model, directory):
    msg = "I should be testing model {} with data from directory {}"
    print(msg.format(model, directory))   

@main.command("infer")
@click.option('-m', '--model')
@click.option('-d', '--directory')
def infer(model, directory):
    msg = "I should be infering using model {} with data from directory {}"
    print(msg.format(model, directory))

    
if __name__ == '__main__':
    main(obj={})