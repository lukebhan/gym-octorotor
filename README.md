**Gym Octorotor**

This is a Octorotor fully compatible with OpenAI gym. For now, it allows you to provide your own controllers along with your own motor and motor controller. 

**Installation**

```
git clone [https://github.com/lukebhan/gym-octorotor.git](https://github.com/lukebhan/gym-octorotor.git)

pip install -e .
```

You will also need

- zlib

**Examples**
Currently there are two examples that both used physics based PID controllers. 
The first example is just an altitude controller that plots the altitude according to some Zref.

The second controller actually runs the openai environment and goes to some ref x, y point. 

More examples will be added!
