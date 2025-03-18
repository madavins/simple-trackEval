import os
import numpy as np

def plot_results(results, output_dir, tracker_name, plots_list=None):
    """
    Creates plots of results for a single tracker.
    Args:
        results: Dictionary of results from the metric.eval_sequence calls.
        output_dir: Where to save the plots.
        tracker_name: name of the tracker for the title.
        plots_list: List of plots to generate.
    """
    if plots_list is None:
        plots_list = get_default_plots_list()

    for args in plots_list:
        create_single_plot(results, output_dir, tracker_name, *args)


def get_default_plots_list():
    # y_label, x_label, sort_label, bg_label, bg_function
    plots_list = [
        ['AssA', 'DetA', 'HOTA', 'HOTA', 'geometric_mean'],
        ['AssPr', 'AssRe', 'HOTA', 'AssA', 'jaccard'],
        ['DetPr', 'DetRe', 'HOTA', 'DetA', 'jaccard'],
        ['HOTA(0)', 'LocA(0)', 'HOTA', 'HOTALocA(0)', 'multiplication'],
        ['HOTA', 'LocA', 'HOTA', None, None],
        ['HOTA', 'MOTA', 'HOTA', None, None],
        ['HOTA', 'IDF1', 'HOTA', None, None],
        ['IDF1', 'MOTA', 'HOTA', None, None],
    ]
    return plots_list


def create_single_plot(results, output_dir, tracker_name, y_label, x_label, sort_label, bg_label=None, bg_function=None, settings=None):
    """Creates a single plot (adapted from create_comparison_plot)."""

    from matplotlib import pyplot as plt

    if settings is None:
        gap_val = 2
    else:
        gap_val = settings['gap_val']

    if (bg_label is None) != (bg_function is None):
        raise Exception('bg_function and bg_label must either be both given or neither given.')

    # Access results directly, no loading needed.
    if sort_label not in results:
        return # Metric not computed.
    if x_label not in results[sort_label] or y_label not in results[sort_label]:
        return

    x_values = np.array([results[sort_label][x_label]])
    y_values = np.array([results[sort_label][y_label]])
    if len(x_values.shape) > 1:
        x_values = x_values[0]
        y_values = y_values[0]


    # Find best fitting boundaries for data
    boundaries = _get_boundaries(x_values, y_values, round_val=gap_val/2)

    fig = plt.figure()

    if bg_function is not None:
        _plot_bg_contour(bg_function, boundaries, gap_val, results)

    # Plot data points
    plt.plot(x_values, y_values, 'b.', markersize=15)

    # Add extra explanatory text to plots
    if bg_label is not None:
        plt.text(1, -0.11, 'curve values:\n' + bg_label, horizontalalignment='right', verticalalignment='center',
                 transform=fig.axes[0].transAxes, color="grey", fontsize=12)

    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    title = y_label + ' vs ' + x_label
    if bg_label is not None:
        title += ' (' + bg_label + ')'
    plt.title(title, fontsize=17)
    plt.xticks(np.arange(0, 100, gap_val))
    plt.yticks(np.arange(0, 100, gap_val))
    min_x, max_x, min_y, max_y = boundaries
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, tracker_name + '_' + title.replace(' ', '_'))
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def _get_boundaries(x_values, y_values, round_val):
    x1 = np.min(np.floor((x_values - 0.5) / round_val) * round_val)
    x2 = np.max(np.ceil((x_values + 0.5) / round_val) * round_val)
    y1 = np.min(np.floor((y_values - 0.5) / round_val) * round_val)
    y2 = np.max(np.ceil((y_values + 0.5) / round_val) * round_val)
    x_range = x2 - x1
    y_range = y2 - y1
    max_range = max(x_range, y_range)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    min_x = max(x_center - max_range / 2, 0)
    max_x = min(x_center + max_range / 2, 100)
    min_y = max(y_center - max_range / 2, 0)
    max_y = min(y_center + max_range / 2, 100)
    return min_x, max_x, min_y, max_y


def geometric_mean(x, y):
    return np.sqrt(x * y)


def jaccard(x, y):
    """Calculate Jaccard index with handling for edge cases."""
    x = np.array(x) / 100
    y = np.array(y) / 100
    
    # Avoid division by zero
    denominator = x + y - x * y
    # Where denominator is 0, set it to 1 to avoid division by zero
    denominator = np.where(denominator == 0, 1, denominator)
    
    result = 100 * (x * y) / denominator
    # Replace any NaN values with 0
    result = np.nan_to_num(result, 0)
    return result


def multiplication(x, y):
    return x * y / 100


bg_function_dict = {
    "geometric_mean": geometric_mean,
    "jaccard": jaccard,
    "multiplication": multiplication,
}


def _plot_bg_contour(bg_function, plot_boundaries, gap_val, results):
    """ Plot background contour. """
    from matplotlib import pyplot as plt

    # Plot background contour
    min_x, max_x, min_y, max_y = plot_boundaries
    x = np.arange(min_x, max_x, 0.1)
    y = np.arange(min_y, max_y, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)
    
    try:
        if bg_function in bg_function_dict.keys():
            z_grid = bg_function_dict[bg_function](x_grid, y_grid)
        else:
            raise Exception("background plotting function '%s' is not defined." % bg_function)
            
        levels = np.arange(0, 100, gap_val)
        con = plt.contour(x_grid, y_grid, z_grid, levels, colors='grey')

        def bg_format(val):
            s = '{:1f}'.format(val)
            return '{:.0f}'.format(val) if s[-1] == '0' else s

        con.levels = [bg_format(val) for val in con.levels]
        plt.clabel(con, con.levels, inline=True, fmt='%r', fontsize=8)

        # Plot pareto optimal lines
        if 'HOTA' in results:
            det_a = np.array(results['HOTA']['DetA'], dtype=float)
            ass_a = np.array(results['HOTA']['AssA'], dtype=float)
            
            # Ensure we have valid numeric data
            if np.isfinite(det_a).all() and np.isfinite(ass_a).all():
                _plot_pareto_optimal_lines(det_a, ass_a)
                
    except Exception as e:
        pass


def _plot_pareto_optimal_lines(x_values, y_values):
    """ Plot pareto optimal lines """
    from matplotlib import pyplot as plt
    
    x_values = x_values.flatten()
    y_values = y_values.flatten()
    
    if len(x_values) == 0 or len(y_values) == 0:
        return

    # Plot pareto optimal lines
    cxs = x_values
    cys = y_values
    
    # Initialize pareto lines
    best_y = np.argmax(cys)
    
    if best_y >= len(cxs):
        return
        
    x_pareto = [0, cxs[best_y]]
    y_pareto = [cys[best_y], cys[best_y]]
    t = 2
    
    remaining = cxs > x_pareto[t - 1]
    cys = cys[remaining]
    cxs = cxs[remaining]
    
    while len(cxs) > 0 and len(cys) > 0:
        best_y = np.argmax(cys)
        if best_y >= len(cxs):
            break
        x_pareto += [x_pareto[t - 1], cxs[best_y]]
        y_pareto += [cys[best_y], cys[best_y]]
        t += 2
        remaining = cxs > x_pareto[t - 1]
        cys = cys[remaining]
        cxs = cxs[remaining]
    
    if len(x_pareto) > 0:
        x_pareto.append(x_pareto[-1])
        y_pareto.append(0)
    
    plt.plot(np.array(x_pareto), np.array(y_pareto), '--r')