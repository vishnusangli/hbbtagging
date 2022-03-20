import yaml
import matplotlib.pyplot as plt

class Formatter:
    """
    Obtain and apply formatting information using presets in a
    configuration file.

    Each preset has a name (`name` parmeter in all functions) and s series of
    properties accessed by `key`. They properties can be nested, in which case
    they `key` is a list (`['level0','level1']`) or a `/` separated string
    (`'level0/level1'`).

    The generic parmeters are returned using the `parameter` function. There are
    a few helper functions that use specific parameters to apply formatting of
    objects. Some return `kwargs` for common functions (ie: `hist`) and some
    modify objects (ie: `xlabel` modifies MPL subplot).
    
    A few common parameters are:
    - `fig/*`: styling of a figure
    - `hist/*`: creatino of a histogram

    """
    def __init__(self, path):
        """
        Load presets from a YAML file located at `path`.

        The format of the YAML file is:
        ```yaml
        name0:
            key0: value0
        name1:
            key0: value0
        ```
        """
        self.data={}
        with open(path,'r') as f:
            self.data=yaml.safe_load(f)

    def parameter(self, name, key, default=None):
        """
        Obtain parameter `key` from `name` preset. If the preset does not exist,
        then `default` is returned.
        """
        if name not in self.data:
            return default
        mydata=self.data[name]

        # Format for recursive navigation
        if type(key) is str:
            key=key.split('/')

        # Find the value
        for k in key:
            if k not in mydata:
                return default
            mydata=mydata[k]
        return mydata

    def hist(self, name, bins=None, range=None):
        """
        Get arguments for making a histogram of variable `name`.

        The following keys are used:
         - `hist/bins`: the number of bins
         - `hist/range`: the range of the bins
        """
        args={'histtype':'step'}
        args['bins' ]=self.parameter(name, 'hist/bins' , bins )
        args['range']=self.parameter(name, 'hist/range', range)
        return args

    def subplot(self, name, ax=None):
        """
        Apply all available parameters to subplot `ax`.
        """
        
        if ax is None: ax=plt.gca()
        self.xlabel(name, ax=ax)

    def xlabel(self, name, ax=None):
        """
        Set the xlabel for subplot `ax` as `fig/label [fig/units]`.

        If `fig/units` are `None`, then no `[...]` is added
        """
        if ax is None: ax=plt.gca()
        label=self.parameter(name, 'fig/label', '')
        units=self.parameter(name, 'fig/units', None)
        if units!=None:
            label=f'{label} [{units}]'
        ax.set_xlabel(label)