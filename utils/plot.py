import numpy                as np
import matplotlib.pyplot    as plt
import pandas               as pd

def accu_2_int(df:pd.DataFrame,
               column:str='avg_train_acc_nopad_till_current_batch'):
    Dump = []
    for index in range(len(df)):
        Dump.append(float(np.array(df[column])[index][7:-18]))
    return Dump

def result_plot(model_name:str,
             plot_desc:str,
             data_1:list,
             data_2:list,
             title : str,
             DPI=100,
             axis_label_size=15,
             x_grid=0.1,
             y_grid=0.5,
             axis_size=12,
             y_lim =[0,6]):
    
    assert(len(data_1)==len(data_2))
    
    '''
        in this function by getting the two list of data result will be ploted as the 
        TA desired
        
        Parameters
        ----------
        model_name:str : dosent do any thing in this vesion but can implemented to save image file
            with the given name
        
        plot_desc:str : define the identity of the data i.e. Accuracy , loss ,...
        
        data_1:list     : list of first data set .
        data_2:list     : list of second data set .
        
        optional
        
        DPI=100             :   define the quality of the plot
        axis_label_size=15  :   define the label size of the axises
        x_grid=0.1          :   x_grid capacity
        y_grid=0.5          :   y_grid capacity
        axis_size=12        :   axis number's size
        
        
        Returns
        -------
        none      : by now 
        

        See Also
        --------
        size of the two list must be same 

        Notes
        -----
        size of the two list must be same 

        Examples
        --------
        >>> result_plot("perceptron",
             "loss",
             data_1,
             data_2)

        '''
    
    fig , ax = plt.subplots(1,figsize=(10,5) , dpi=DPI)

    fig.suptitle(f"Train and validation "+plot_desc,y=0.95 , fontsize=20)

    epochs = range(len(data_1))
    ax.plot(epochs, data_1, 'b',linewidth=3, label='tarin '+ plot_desc)
    ax.plot(epochs, data_2, 'r',linewidth=3, label='validation '+plot_desc)

    ax.set_xlabel("epoch"       ,fontsize=axis_label_size)
    ax.set_ylabel(plot_desc     ,fontsize=axis_label_size)
    if x_grid:
        ax.grid(axis="x",alpha=0.1)
    if y_grid:
        ax.grid(axis="y",alpha=0.5)
    
    ax.set_title(title, pad=40)

    ax.legend(loc=0,prop={"size":9})

    ax.tick_params(axis="x",labelsize=axis_size)
    ax.tick_params(axis="y",labelsize=axis_size)

    #spine are borde line of plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    
    # ax.set_ylim([8.48,8.6])
    ax.set_ylim(y_lim)
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()
    
