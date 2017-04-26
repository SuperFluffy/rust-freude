use syn::{
    self,
    MetaItem,
    NestedMetaItem,
};

#[derive(Debug)]
pub struct Container<'a> {
    pub ident: syn::Ident,
    pub attrs: ContainerAttributes,
    pub fields: Vec<Field<'a>>,
    pub generics: &'a syn::Generics,
}

#[derive(Clone, Debug)]
pub struct Field<'a> {
    pub ident: Option<syn::Ident>,
    pub attrs: FieldAttributes,
    pub ty: &'a syn::Ty,
}

/// Represents container (e.g. struct) attribute information
#[derive(Debug)]
pub struct ContainerAttributes {
    pub after_integration: Option<syn::Path>,
    pub after_step: Option<syn::Path>,
}

#[derive(Debug)]
struct Attr<T> {
    name: &'static str,
    value: Option<T>,
}

/// Represents field attribute information
#[derive(Clone, Debug)]
pub struct FieldAttributes {
    pub state: bool,
    pub stepper: bool,
    pub system: bool,
}

#[derive(Debug)]
struct BoolAttr(Attr<()>);

impl<'a> Container<'a> {
    pub fn from_ast(item: &'a syn::DeriveInput) -> Container<'a> {
        let attrs = ContainerAttributes::from_ast(item);

        let err_msg = "Deriving Integrator is only supported on standard (non-tuple, non-unit) structs.";
        let fields = match item.body {
            syn::Body::Struct(ref variant_data) => {
                match variant_data {
                    &syn::VariantData::Struct(ref fields) => fields,
                    _ => panic!(err_msg),
                }
            },
            _ => panic!(err_msg),
        };

        let fields = fields_from_ast(fields);

        let item = Container {
            ident: item.ident.clone(),
            attrs: attrs,
            fields: fields,
            generics: &item.generics,
        };
        item
    }
}

impl ContainerAttributes {
  /// Extract out the `#[freude(...)]` attributes from the struct.
    pub fn from_ast(item: &syn::DeriveInput) -> Self {
        let mut after_integration = Attr::none("after_integration");
        let mut after_step = Attr::none("after_step");

        for meta_items in item.attrs.iter().filter_map(get_freude_meta_items) {
            for meta_item in meta_items {
                match meta_item {
                    NestedMetaItem::MetaItem(MetaItem::NameValue(ref ident, ref lit)) if ident == "after_integration" => {
                        if let Ok(path) = parse_lit_into_path(ident.as_ref(), lit) {
                            match item.body {
                                syn::Body::Struct(syn::VariantData::Struct(_)) => {
                                    after_integration.set(path);
                                },
                                _ => panic!("#[freude(after_integration = \"...\")] can only be used on structs with named fields")
                            }
                        }
                    }

                    NestedMetaItem::MetaItem(MetaItem::NameValue(ref ident, ref lit)) if ident == "after_step" => {
                        if let Ok(path) = parse_lit_into_path(ident.as_ref(), lit) {
                            match item.body {
                                syn::Body::Struct(syn::VariantData::Struct(_)) => {
                                    after_step.set(path);
                                },
                                _ => panic!("#[freude(after_step = \"...\")] can only be used on structs with named fields")
                            }
                        }
                    }

                    NestedMetaItem::MetaItem(ref meta_item) => {
                        panic!("Unknown freude container attribute `{}`", meta_item.name());
                    }

                    NestedMetaItem::Literal(_) => {
                        panic!("unexpected literal in serde container attribute");
                    }
                }
            }
        }

        ContainerAttributes {
            after_integration: after_integration.get(),
            after_step: after_step.get(),
        }
    }
}

impl FieldAttributes {
    /// Extract out the `#[freude(...)]` attributes from a struct field.
    pub fn from_ast(field: &syn::Field) -> Self {
        let mut state = BoolAttr::none("state");
        let mut stepper = BoolAttr::none("stepper");
        let mut system = BoolAttr::none("system");

        for meta_items in field.attrs.iter().filter_map(get_freude_meta_items) {
            for meta_item in meta_items {
                match meta_item {
                    // Parse `#[freude(state)]`
                    NestedMetaItem::MetaItem(MetaItem::Word(ref ident)) if ident == "state" => {
                        state.set_true();
                    }

                    NestedMetaItem::MetaItem(MetaItem::Word(ref ident)) if ident == "stepper" => {
                        stepper.set_true();
                    }

                    NestedMetaItem::MetaItem(MetaItem::Word(ref ident)) if ident == "system" => {
                        system.set_true();
                    }

                    NestedMetaItem::MetaItem(ref meta_item) => {
                        panic!("unknown freude field attribute `{}`", meta_item.name());
                    }

                    NestedMetaItem::Literal(_) => {
                        panic!("unexpected literal in freude field attribute");
                    }
                }
            }
        }

        FieldAttributes {
            state: state.get(),
            stepper: stepper.get(),
            system: system.get(),
        }
    }
}

impl<T> Attr<T> {
    fn none(name: &'static str) -> Self {
        Attr {
            name: name,
            value: None,
        }
    }

    fn set(&mut self, value: T) {
        if self.value.is_some() {
            panic!("duplicate freude attribute `{}`", self.name);
        } else {
            self.value = Some(value);
        }
    }

    fn set_opt(&mut self, value: Option<T>) {
        if let Some(value) = value {
            self.set(value);
        }
    }

    fn set_if_none(&mut self, value: T) {
        if self.value.is_none() {
            self.value = Some(value);
        }
    }

    fn get(self) -> Option<T> {
        self.value
    }
}

impl BoolAttr {
    fn none(name: &'static str) -> Self {
        BoolAttr(Attr::none(name))
    }

    fn set_true(&mut self) {
        self.0.set(());
    }

    fn get(&self) -> bool {
        self.0.value.is_some()
    }
}

fn fields_from_ast<'a>(fields: &'a [syn::Field]) -> Vec<Field<'a>> {
    fields
        .iter()
        .map(
            |field| {
                Field {
                    ident: field.ident.clone(),
                    attrs: FieldAttributes::from_ast(field),
                    ty: &field.ty,
                }
            },
        )
        .collect()
}

pub fn get_freude_meta_items(attr: &syn::Attribute) -> Option<Vec<NestedMetaItem>> {
    match attr.value {
        MetaItem::List(ref name, ref items) if name == "freude" => Some(items.iter().cloned().collect()),
        _ => None,
    }
}

fn get_string_from_lit(
    attr_name: &str,
    meta_item_name: &str,
    lit: &syn::Lit,
) -> Result<String, String> {
    if let syn::Lit::Str(ref s, _) = *lit {
        Ok(s.clone())
    } else {
        Err(format!(
                "expected serde {} attribute to be a string: `{} = \"...\"`",
                attr_name,
                meta_item_name
            )
        )
    }
}

fn parse_lit_into_path(attr_name: &str, lit: &syn::Lit) -> Result<syn::Path, String> {
    let string = try!(get_string_from_lit(attr_name, attr_name, lit));
    syn::parse_path(&string).map_err(|err| err.to_string())
}
