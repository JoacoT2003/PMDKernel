using GLMakie, Printf

"""
    mostrar_grilla(B, grid_range; zoom=3)

Visualiza el campo computado por `B0(R, P, M)` con un Slicer3D ortogonal.

Asume que `R` se construyó con `geom_grilla_xyz(:nombre, grid_range, grid_range, grid_range).R`,
es decir una grilla cúbica con el mismo `grid_range` en los tres ejes.

`B` es la matriz `[n, 3]` que devuelve `B0` (Tesla). Toma la componente By,
la pasa a mT y muestra (con signo invertido para el colormap).
"""
function mostrar_grilla(B::AbstractMatrix, grid_range::AbstractRange; zoom::Int = 3)
    Nx = length(grid_range); Ny = Nx; Nz = Nx
    @assert size(B, 1) == Nx*Ny*Nz "mostrar_grilla: tamaño de B inconsistente con grid_range"
    By = reshape(B[:, 2] .* -1000f0, Nx, Ny, Nz)   # mT, signo invertido para colormap
    fig = Figure(size=(600, 600))
    Slicer3D(fig, By, zoom=zoom)
    display(fig)
    return fig
end

function Slicer3D(fig,data;
                        colormap=:viridis,colorrange=nothing,
                        zoom::Int=1,
                        haircross=true,pointvalue=true)

    debug = true

    # Layout:
    # Data has dimensions B[x,y,z] x: R->L y: P->A z: F->H
    # --------------------------------
    # |    empty   | y  axial  |     |
    # |            |      x    |sl y |
    # |------------------------|-----|
    # | z sagittal | z coronal |     |
    # |      y     |     x     |sl z |
    # --------------------------------
    # |            |   sl x    |     |
    # --------------------------------

    # Obtain parameters from data
    sizesag,sizecor,sizeaxi = size(data)
    zsizesag,zsizecor,zsizeaxi = zoom.*size(data)
    startpoint = Point3(Int.(round.(size(data)./2)))
    if isnothing(colorrange) # Problem with NaN
        crange = extrema(filter(!isnan,data))
        # (minimum(data),maximum(data))
        # if isnan(crange[1])||isnan(crange[2])
        #     println("NaN in data, range set to (-1,1)")
        #     crange = (-1,1)
        # end
    else
        crange = colorrange
    end
    if debug
        println("size(data) = $(size(data))")
        println("crange = $crange")
    end

    # Create panels and sliders
        # fig = Figure(size=(650,500)) # Need to control the size
        aaxi = Axis(fig[1,2],aspect=DataAspect(),height=zsizecor,width=zsizesag)
        asag = Axis(fig[2,1],aspect=DataAspect(),height=zsizeaxi,width=zsizecor)
        acor = Axis(fig[2,2],aspect=DataAspect(),height=zsizeaxi,width=zsizesag)
        hidespines!(aaxi); hidedecorations!(aaxi)
        hidespines!(acor); hidedecorations!(acor)
        hidespines!(asag); hidedecorations!(asag)
        lpvalue = Label(fig[3,1],"")
        # Need to get rid of label (or position it in a better way)
        saxi = SliderGrid(fig[2,3],
                    (range=1:sizeaxi,startvalue=startpoint[3],horizontal=false,height=zsizeaxi))
        ssag = SliderGrid(fig[3,2],
                (range=1:sizesag,startvalue=startpoint[1],horizontal=true,width=zsizesag))
        scor = SliderGrid(fig[1,3],
            (range=1:sizecor,startvalue=startpoint[2],horizontal=false,height=zsizecor))

    # The interactive part
    @lift begin
        x = $(ssag.sliders[1].value)
        y = $(scor.sliders[1].value)
        z = $(saxi.sliders[1].value)
        heatmap!(asag,data[x,:,:],colormap=colormap,colorrange=crange)
        heatmap!(acor,data[:,y,:],colormap=colormap,colorrange=crange)
        heatmap!(aaxi,data[:,:,z],colormap=colormap,colorrange=crange)
        if haircross
            lines!(aaxi,[1;sizesag],[y;y],color=:white)
            lines!(aaxi,[x;x],[1;sizecor],color=:white)
            lines!(acor,[1;sizesag],[z;z],color=:white)
            lines!(acor,[x;x],[1;sizeaxi],color=:white)
            lines!(asag,[1;sizecor],[z;z],color=:white)
            lines!(asag,[y;y],[1;sizeaxi],color=:white)
        end
        if pointvalue
            lpvalue.text = @sprintf("(%d,%d,%d) -> %.4f",
                                    x,y,z,data[x,y,z])
        end
    end

    return saxi # returns the upper left grid to be used by the user

    # display(fig)

end