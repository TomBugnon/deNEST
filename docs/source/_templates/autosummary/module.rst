{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

{% if modules %}
.. rubric:: Modules
.. autosummary::
    :toctree:
    {% for modules in modules %}
    {{ modules }}
    {% endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes
.. autosummary::
    :toctree:
    {% for class in classes %}
    {{ class }}
    {% endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions
.. autosummary::
    :toctree:
    {% for function in functions %}
    {{ function }}
    {% endfor %}
{% endif %}