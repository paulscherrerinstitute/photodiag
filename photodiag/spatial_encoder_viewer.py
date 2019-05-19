import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Span
from bokeh.plotting import figure
from IPython.display import display
from ipywidgets import IntRangeSlider, IntSlider, Layout

from photodiag.spatial_encoder import SpatialEncoder

output_notebook()


class SpatialEncoderViewer(SpatialEncoder):
    def plot_hdf5(self, filepath):
        """Experimental viewer for hdf5 files in a jupyter notebook.

        Args:
            filepath: hdf5 file to be processed
        """
        results = self.process_hdf5(filepath, debug=True)

        images = results['images']
        images_proj = images.mean(axis=1)

        edge_pos = results['edge_pos']
        xcorr_data = results['xcorr']
        orig_data = results['raw_input']

        source_im = ColumnDataSource(
            data=dict(
                image=[images[0]],
                x=[-0.5],
                y=[self.roi[0]],
                dw=[images.shape[2]],
                dh=[self.roi[1] - self.roi[0]],
            )
        )

        data_len = orig_data.shape[1]
        source_orig = ColumnDataSource(
            data=dict(
                x=np.arange(data_len), y=orig_data[0], y_bkg=self._background, y_proj=images_proj[0]
            )
        )

        xcorr_len = xcorr_data.shape[1]
        source_xcorr = ColumnDataSource(
            data=dict(
                x=np.arange(xcorr_len) * self.refinement + np.floor(self.step_length / 2),
                y=xcorr_data[0],
            )
        )

        p_im = figure(
            height=200,
            width=800,
            title='Camera ROI Image',
            x_range=(0, images.shape[2]),
            y_range=self.roi,
        )
        p_im.image(
            image='image', x='x', y='y', dw='dw', dh='dh', source=source_im, palette='Viridis256'
        )

        p_nobkg = figure(height=200, width=800, title='Projection and background')
        p_nobkg.line('x', 'y_bkg', source=source_orig, line_color='black')
        p_nobkg.line('x', 'y_proj', source=source_orig)
        p_nobkg.x_range = p_im.x_range

        p_orig = figure(height=200, width=800, title='Background removed')
        p_orig.line('x', 'y', source=source_orig)
        p_orig.x_range = p_im.x_range

        p_xcorr = figure(height=200, width=800, title='Xcorr')
        p_xcorr.line('x', 'y', source=source_xcorr)
        p_xcorr.x_range = p_im.x_range

        span_args = dict(dimension='height', line_color='red')
        if np.isnan(edge_pos[0]):
            span_args['location'] = 0
            span_args['visible'] = False
        else:
            span_args['location'] = edge_pos[0]

        s_im = Span(**span_args)
        p_im.add_layout(s_im)

        s_nobkg = Span(**span_args)
        p_nobkg.add_layout(s_nobkg)

        s_orig = Span(**span_args)
        p_orig.add_layout(s_orig)

        s_xcorr = Span(**span_args)
        p_xcorr.add_layout(s_xcorr)

        layout = gridplot(
            [p_im, p_nobkg, p_orig, p_xcorr], ncols=1, toolbar_options=dict(logo=None)
        )

        handle = show(layout, notebook_handle=True)

        # Slider
        def slider_callback(change):
            new = change['new']
            source_im.data.update(image=[images[new]])

            source_orig.data.update(y=orig_data[new], y_proj=images_proj[new])
            source_xcorr.data.update(y=xcorr_data[new])

            if np.isnan(edge_pos[new]):
                s_im.visible = False
                s_nobkg.visible = False
                s_orig.visible = False
                s_xcorr.visible = False
            else:
                s_im.visible = True
                s_nobkg.visible = True
                s_orig.visible = True
                s_xcorr.visible = True
                s_im.location = edge_pos[new]
                s_nobkg.location = edge_pos[new]
                s_orig.location = edge_pos[new]
                s_xcorr.location = edge_pos[new]

            push_notebook(handle=handle)

        slider = IntSlider(
            min=0,
            max=len(edge_pos) - 1,
            value=0,
            step=1,
            description="Shot",
            continuous_update=False,
            layout=Layout(width='800px'),
        )

        slider.observe(slider_callback, names='value')
        display(slider)

    def plot_calibrate_time(self, *args, **kwargs):
        scan_pos_fs, edge_pos_pix, fit_coeff = self.calibrate_time(*args, **kwargs)

        source_results = ColumnDataSource(data=dict(x=scan_pos_fs, y=edge_pos_pix))

        source_fit = ColumnDataSource(
            data=dict(
                x=[scan_pos_fs[0], scan_pos_fs[-1]],
                y=np.polyval(fit_coeff, [scan_pos_fs[0], scan_pos_fs[-1]]),
            )
        )

        p_time = figure(
            height=400,
            width=800,
            title='Time calibration',
            x_axis_label='Stage position, fs',
            y_axis_label='Edge position, pix',
        )
        p_time.scatter('x', 'y', source=source_results)
        p_time.line('x', 'y', line_color='red', source=source_fit)

        layout = gridplot([p_time], ncols=1, toolbar_options=dict(logo=None))

        handle = show(layout, notebook_handle=True)

        # Slider
        def slider_callback(change):
            left = change['new'][0]
            right = change['new'][1]
            fit_coeff = np.polyfit(scan_pos_fs[left : right + 1], edge_pos_pix[left : right + 1], 1)
            self.pix_per_fs = fit_coeff[0]

            source_fit.data.update(
                x=[scan_pos_fs[left], scan_pos_fs[right]],
                y=np.polyval(fit_coeff, [scan_pos_fs[left], scan_pos_fs[right]]),
            )

            push_notebook(handle=handle)

        slider = IntRangeSlider(
            min=0,
            max=len(scan_pos_fs) - 1,
            value=[0, len(scan_pos_fs) - 1],
            step=1,
            description="Fit range",
            continuous_update=False,
            layout=Layout(width='800px'),
        )

        slider.observe(slider_callback, names='value')
        display(slider)
