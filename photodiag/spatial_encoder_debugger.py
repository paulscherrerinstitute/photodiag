import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Span, Slider
from bokeh.plotting import figure, output_notebook, show
from photodiag.spatial_encoder import SpatialEncoder

output_notebook()


class SpatialEncoderDebugger(SpatialEncoder):
    def plot_hdf5(self, filepath, port=8888):
        def modify_doc(doc):
            results = self.process_hdf5(filepath, debug=True)
            images = self._read_bsread_image(filepath)
            images_proj = images.mean(axis=1)

            edge_pos = results['edge_pos']
            xcorr_data = results['xcorr']
            orig_data = results['raw_input']

            source_im = ColumnDataSource(
                data=dict(
                    image=[images[0]],
                    x=[0],
                    y=[0],
                    dw=[1000],
                    dh=[100],
                )
            )

            data_len = np.shape(orig_data)[1]
            source_orig = ColumnDataSource(
                data=dict(
                    x=np.arange(data_len),
                    y=orig_data[0],
                    y_bkg=self._background,
                    y_proj=images_proj[0],
                )
            )

            xcorr_len = np.shape(xcorr_data)[1]
            source_xcorr = ColumnDataSource(
                data=dict(
                    x=np.arange(xcorr_len) + np.floor(self.step_length/2),
                    y=xcorr_data[0],
                )
            )

            p_im = figure(
                height=200, width=800, title='Camera ROI Image',
                x_range=(0, 1000), y_range=(0, 100),
            )
            p_im.image(
                image='image', x='x', y='y', dw='dw', dh='dh', source=source_im,
                palette='Viridis256',
            )
            s_im = Span(location=edge_pos[0], dimension='height', line_color='red')
            p_im.add_layout(s_im)

            p_nobkg = figure(height=200, width=800, title='Projection and background')
            p_nobkg.line('x', 'y_bkg', source=source_orig, line_color='black')
            p_nobkg.line('x', 'y_proj', source=source_orig)
            s_nobkg = Span(location=edge_pos[0], dimension='height', line_color='red')
            p_nobkg.add_layout(s_nobkg)

            p_nobkg.x_range = p_im.x_range

            p_orig = figure(height=200, width=800, title='No background data')
            p_orig.line('x', 'y', source=source_orig)
            s_orig = Span(location=edge_pos[0], dimension='height', line_color='red')
            p_orig.add_layout(s_orig)

            p_orig.x_range = p_im.x_range

            p_xcorr = figure(height=200, width=800, title='Xcorr')
            p_xcorr.line('x', 'y', source=source_xcorr)
            s_xcorr = Span(location=edge_pos[0], dimension='height', line_color='red')
            p_xcorr.add_layout(s_xcorr)

            p_xcorr.x_range = p_im.x_range

            def slider_callback(_attr, _old, new):
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

            slider = Slider(start=0, end=len(edge_pos), value=0, step=1, title="Shot")
            slider.on_change('value', slider_callback)

            layout = gridplot(
                [p_im, p_nobkg, p_orig, p_xcorr, slider],
                ncols=1, toolbar_options=dict(logo=None),
            )

            doc.add_root(layout)

        show(modify_doc, notebook_url="http://localhost:{}".format(port))
