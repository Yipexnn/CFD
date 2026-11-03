# vcpkg overlay ports

- `vtk`: the vcpkg `vtk` port (baseline af839a44) with `qtdeclarative` removed from the `qt`
  feature. The port already disables the Qt Quick module when the QML tools are absent, and
  cfd_gui only needs the widget-based `GUISupportQt`. This avoids building Qt Declarative,
  Qt ShaderTools, Qt SVG and Qt LanguageServer (several hours on CI).

Used by the `windows-msvc-gui-*` presets via `VCPKG_OVERLAY_PORTS`.
